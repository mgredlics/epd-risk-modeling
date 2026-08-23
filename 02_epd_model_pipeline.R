# --- EPD (Early Payment Default) Model---
# This Workflow builds an account-level model to estimate the probability of 
# early payment default and generate monthly confidence intervals around
# the expected EPD rate. These outputs helps collections teams plan resources 
# by giving insight into future trends

# ---- MLflow setup ----
# When running locally, I start the MLflow server in PowerShell using:
# C:\Users\mgred\AppData\Local\Programs\Python\Python314\python.exe -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri "sqlite:///C:/mlflow_local/mlflow.db" --default-artifact-root "file:///C:/mlflow_local/artifacts"
# The MLflow UI can be found at: http://127.0.0.1:5000/

# --- Libraries ---
# Core Libraries for tidymodels modelling

library(dplyr)
library(readr)
library(lubridate)
library(stringr)

library(tidymodels)
library(scorecard)
library(monobin)
library(yardstick)
library(probably)

library(glmnet)
library(caret)
library(ROCR)
library(vip)

library(mlflow)
library(future)

tidymodels_prefer()
options(digits = 8)

'%notin%' <- function(x,y) !(x %in% y)


# ---- MLflow environment configuration ----
# I’m running this workflow locally, so I explicitly set the tracking URI and Python binary.

Sys.unsetenv("MLFLOW_BIN")
Sys.setenv(MLFLOW_TRACKING_URI = "http://127.0.0.1:5000")

Sys.setenv(
  MLFLOW_PYTHON_BIN = "C:/Users/mgred/AppData/Local/Programs/Python/Python314/python.exe"
)

# Quick sanity check — if the tracking URI isn't set, MLflow falls back to defaults
if (Sys.getenv("MLFLOW_TRACKING_URI") == "") {
  stop("MLFLOW_TRACKING_URI is not set. Check ~/.Renviron and restart RStudio.")
}

# Increase the timeout to 600 to avoid false failures
Sys.setenv(MLFLOW_HTTP_REQUEST_TIMEOUT = "600")


# ---- Experiment metadata ----
month_tag <- "Aug2026"
local_folder <- "C:/Users/mgred/OneDrive/Documents/Data Science/EarlyDefault/"
mlflow_set_experiment("EPD_model_LocalRun")

# ---- MLflow logging helpers ----
# Small utilities to help keep artifact logging consistent and readable

mlflow_log_csv <- function(df, path) {
  readr::write_csv(df, path)
  mlflow_log_artifact(path)
}

mlflow_log_rds <- function(obj, path) {
  saveRDS(obj,path)
  mlflow_log_artifact(path)
}

# MLflow helper to render ggplot to PNG and log file
mlflow_log_plot_png <- function(plot_obj, path, width = 1200, height = 800) {
  grDevices::png(filename=path, width=width, height = height)
  print(plot_obj)
  grDevices::dev.off()
  mlflow_log_artifact(path)
}

# ---- Monte Carlo Simulation ----
# Simulate monthly EPD outcomes using predicted probabilities
# I use 500 draws per month and extract the 5th/50th/95th percentiles for my bands

MonteCarlo_byMonth <- function(scored_df,
                               date_col = "FUNDING_DATE",
                               prob_col = "EPD_Probability",
                               n_sims = 500,
                               seed = 772) {
  stopifnot(date_col %in% names(scored_df))
  stopifnot(prob_col %in% names(scored_df))
  
  set.seed(seed)
  
  scored_df %>%
    dplyr::mutate(
      year_month = format(as.Date(.data[[date_col]]), "%Y-%m"),
      p = .data[[prob_col]]
    ) %>%
    dplyr::filter(!is.na(year_month), !is.na(p)) %>%
    dplyr::group_by(year_month) %>%
    dplyr::summarise(
      count = dplyr::n(),
      expected_percent = mean(p) * 100,
      {
        n <- length(p)
        draws <- matrix(
          stats::rbinom(n * n_sims, size = 1, prob = rep(p, times = n_sims)),
          nrow = n, ncol = n_sims
        )
        sims <- colMeans(draws) * 100
        q <- stats::quantile(sims, probs = c(0.05, 0.50, 0.95), names = FALSE, type = 7)
        dplyr::tibble(p05 = q[1], p50 = q[2], p95 = q[3])
      },
      .groups = "drop"
    ) %>%
    dplyr::arrange(year_month)
}

# ---- KS Function ----
ks_from_roc <- function(data, truth, estimate, event_level = "second") {
  roc_vals <- roc_curve(data, {{ truth }}, {{ estimate }}, event_level = event_level)
  max(roc_vals$sensitivity - (1 - roc_vals$specificity))
}

# ---- Custom recipe step: monobin + WoE transformation ----
# This step combines monotonic binning and WOE conversion into the tidymodels pipeline.
# It is used for bureau variables because monotonicity helps keep the relationship
# with EPD clean and predictable. Thus the model is easier to explain/defend

# The step also handles bureau 'Special Values' (sc) such as -9,-99, etc.
# (Sometimes these values are 99,999, etc. but they get converted to negative numbers earlier in process)

step_monobin_woe <- function(recipe,
                             ...,
                             outcome,
                             role = "predictor",   
                             sc = c(NA, NaN, Inf, -Inf, -9:-1, -99:-91),
                             sc.method = "together",
                             y.type = "bina",
                             woe.gap = 0.1,
                             bin_num_limit = 12, 
                             special_values = "use_sc",
                             keep_original_cols = FALSE,
                             trained = FALSE,
                             columns = NULL,
                             bins = NULL,
                             breaks_list = NULL,
                             skip = FALSE,
                             id = recipes::rand_id("monobin_woe")) {
  
  recipes::add_step(
    recipe,
    step_monobin_woe_new(
      terms = rlang::enquos(...),
      outcome = rlang::as_name(rlang::ensym(outcome)),
      role = role,          
      trained = trained,
      sc = sc,
      sc.method = sc.method,
      y.type = y.type,
      woe.gap = woe.gap,
      bin_num_limit = bin_num_limit,
      special_values = special_values,
      keep_original_cols = keep_original_cols,
      columns = columns,
      bins = bins,
      breaks_list = breaks_list,
      skip = skip,
      id = id
    )
  )
}

# This helper builds the step object that tidymodels uses internally.
# It stores metadata such as the selected variables, learned bins, and WoE
# dictionary so the step can be applied consistently during baking.

step_monobin_woe_new <- function(terms, outcome, role, trained,
                                 sc, sc.method, y.type, woe.gap,
                                 bin_num_limit, special_values,
                                 keep_original_cols,
                                 columns, bins, breaks_list,
                                 skip, id) {
  
  recipes::step(
    subclass = "monobin_woe",
    terms = terms,
    role = role,
    trained = trained,
    skip = skip,
    id = id,
    outcome = outcome,
    sc = sc,
    sc.method = sc.method,
    y.type = y.type,
    woe.gap = woe.gap,
    bin_num_limit = bin_num_limit,
    special_values = special_values,
    keep_original_cols = keep_original_cols,
    columns = columns,
    bins = bins,
    breaks_list = breaks_list
  )
}

# --- prep(): learn monotonic bins on training data ---
# Creates a frozen set of bins + WoE values that can be applied consistently
# to train/test/OOT.
# For each selected variable:
#   1. Check for enough variation to be binned
#   2. Run monotonic binning (monobin::woe.bin)
#   3. Convert binning output into breakpoints
#   4. Build the WoE dictionary using scorecard::woebin that stays frozen throughout pipeline

prep.step_monobin_woe <- function(x, training, info = NULL, ...) {
  
  # Identify which predictors this step applies to.
  col_names <- recipes::recipes_eval_select(x$terms, training, info)
  
  # Convert outcome to numeric 0/1 for monotonic binning.
  y_raw <- training[[x$outcome]]
  y01 <- if (is.factor(y_raw)) as.numeric(y_raw) - 1 else as.numeric(y_raw)
  
  # Drop variables that are constant or all NA
  usable <- vapply(col_names, function(v) {
    xv <- training[[v]]
    ux <- unique(stats::na.omit(xv))
    length(ux) >= 2
  }, logical(1))
  
  dropped <- col_names[!usable]
  col_names <- col_names[usable]
  
  if (length(dropped) > 0) {
    rlang::warn(paste0("step_monobin_woe: dropped ", length(dropped),
                       " variables with no usable variation: ",
                       paste(dropped, collapse = ", ")))
  }
  
  if (length(col_names) == 0) {
    rlang::abort("step_monobin_woe: no variables available for binning.")
  }
  
  # ---- Run monotonic binning for each variable ----
  mono_tbls <- list()
  
  for (v in col_names) {
    tbl <- tryCatch(
      monobin::woe.bin(
        x = training[[v]],
        y = y01,
        sc = x$sc,
        sc.method = x$sc.method,
        y.type = x$y.type,
        woe.gap = x$woe.gap
      )[[1]],
      error = function(e) NULL
    )
    
    # Keep only valid binning results.
    if (!is.null(tbl) && "x.min" %in% names(tbl) && length(tbl[["x.min"]]) >= 2) {
      mono_tbls[[v]] <- tbl
    }
  }
  
  col_names <- names(mono_tbls)
  
  if (length(col_names) == 0) {
    rlang::abort("step_monobin_woe: all variables failed monotonic binning.")
  }
  
  # ---- Convert binning output into scorecard breakpoints ----
  breaks_list <- lapply(col_names, function(v) {
    tbl <- mono_tbls[[v]]
    
    brks <- tbl[["x.min"]][2:length(tbl[["x.min"]])]
    brks <- brks[!is.na(brks)]
    brks <- brks[!(brks %in% x$sc)]
    brks <- sort(unique(brks))
    
    c("-Inf", as.character(brks))
  })
  names(breaks_list) <- col_names
  
  # ---- Build WoE dictionary ----
  options(scorecard.bin_close_right = FALSE)
  
  sv <- if (identical(x$special_values, "use_sc")) {
    lapply(col_names, function(v) sort(unique(training[[v]][training[[v]] %in% x$sc])))
  } else {
    x$special_values
  }
  names(sv) <- col_names
  
  bins <- scorecard::woebin(
    dt = training %>% dplyr::select(dplyr::all_of(c(x$outcome, col_names))),
    y = x$outcome,
    positive = 1,
    bin_num_limit = x$bin_num_limit,
    breaks_list = breaks_list,
    special_values = sv,
    missing_join = NULL,
    no_cores = 1,
    print_step = 0L
  )
  
  # Neutralize WoE for missing bins.
  for (v in names(bins)) {
    bin_df <- bins[[v]]
    miss_idx <- grepl("missing", bin_df$bin, ignore.case = TRUE)
    if (any(miss_idx)) {
      bin_df$woe[miss_idx] <- 0
      bins[[v]] <- bin_df
    }
  }
  
  # Return trained step object.
  step_monobin_woe_new(
    terms = x$terms,
    outcome = x$outcome,
    role = x$role,
    trained = TRUE,
    sc = x$sc,
    sc.method = x$sc.method,
    y.type = x$y.type,
    woe.gap = x$woe.gap,
    bin_num_limit = x$bin_num_limit,
    special_values = x$special_values,
    keep_original_cols = x$keep_original_cols,
    columns = col_names,
    bins = bins,
    breaks_list = breaks_list,
    skip = x$skip,
    id = x$id
  )
}

#--- bake(): apply frozen bins + WoE to new data
# This is the scoring step that applies the learned bins to the selected columns
bake.step_monobin_woe <- function(object, new_data, ...) {
  
  woe_df <- scorecard::woebin_ply(
    dt = new_data %>% dplyr::select(dplyr::all_of(object$columns)),
    bins = object$bins,
    no_cores = 1, 
    print_step = 0L # Change to 1L if you want additional information
  )
  
  if (!object$keep_original_cols) {
    new_data <- new_data %>% dplyr::select(-dplyr::all_of(object$columns))
  }
  
  dplyr::bind_cols(new_data, woe_df)
}


# --- tidy(): for printing/inspection
# Useful for inspecting the recipe or debuging. It reports whether the step is trainied
# and the included variables

tidy.step_monobin_woe <- function(x, ...) {
  tibble::tibble(
    id = x$id,
    trained = x$trained,
    outcome = x$outcome,
    n_columns = if (is.null(x$columns)) NA_integer_ else length(x$columns)
  )
}

# --- Backtest Functionality
# Function to evaluate how well the model's monthly confidence intervals line up with actuals
# For each month, compare actual EPD results to the 5%/95% intervals
# Count how often actuals fall inside the band (coverage rate)
# Flag the under or over predicting of risk
# Since EPD is lagged, drop the most recent 3 months of data.

Backtest_EPD_Intervals <- function(by_month_actuals,
                                   by_month_mc,
                                   actual_col = "percent",
                                   lower_col  = "p05",
                                   upper_col  = "p95",
                                   expected_coverage = 0.80,
                                   tolerance = 0.05) {
  
  # Identify where to cutoff for the last 3 months
  cutoff_date <- as.Date(format(Sys.Date(), "%Y-%m-01"))
  threshold_date <- seq(cutoff_date, by = "-1 month", length.out = 4)[4]
  
  # Join actuals with Monte Carlo
  bt <- by_month_actuals %>%
    dplyr::inner_join(by_month_mc, by = "year_month") %>%
    dplyr::mutate(year_month_date = as.Date(paste0(year_month, "-01"))) %>%
    dplyr::filter(year_month_date < threshold_date) %>%
    dplyr::mutate(
      actual = .data[[actual_col]],
      lower  = .data[[lower_col]],
      upper  = .data[[upper_col]],
      in_band = if_else(actual >= lower & actual <= upper, 1, 0)
    )
  
  # Coverage rate
  coverage_rate <- mean(bt$in_band, na.rm = TRUE)
  n_periods     <- nrow(bt)
  
  # Confidence interval (90%) for expected coverage
  se_cov <- sqrt(expected_coverage * (1 - expected_coverage) / n_periods)
  lower_ci <- expected_coverage - 1.64 * se_cov #1.64 is z-score for 80%
  upper_ci <- expected_coverage + 1.64 * se_cov #1.64 is z-score for 80%
  
  calibration_flag <- case_when(
    coverage_rate < lower_ci ~ "UNDER-COVERAGE (bands too narrow)",
    coverage_rate > upper_ci ~ "OVER-COVERAGE (bands too wide)",
    TRUE ~ "WELL-CALIBRATED"
  )
  
  # Diagnostics for misses
  bt <- bt %>%
    mutate(
      miss_type = case_when(
        actual < lower ~ "Below Band (underprediction)",
        actual > upper ~ "Above Band (overprediction)",
        TRUE ~ "Inside Band"
      ),
      distance_to_band = case_when(
        actual < lower ~ lower - actual,
        actual > upper ~ actual - upper,
        TRUE ~ 0
      )
    )
  
  miss_summary <- bt %>%
    group_by(miss_type) %>%
    summarise(
      count = n(),
      avg_distance = mean(distance_to_band),
      .groups = "drop"
    )
  
  list(
    summary = tibble::tibble(
      n_periods = n_periods,
      coverage_rate = coverage_rate,
      expected_coverage = expected_coverage,
      lower_ci = lower_ci,
      upper_ci = upper_ci,
      calibration_flag = calibration_flag
    ),
    detail = bt,
    miss_summary = miss_summary
  )
}

#-------------------
#---- MLFlow Run ---
#-------------------

with(mlflow_start_run(), {
  
  # MLFlow names and tags
  run_display_name <- paste0("EPD_GLMNET_", month_tag, "_", format(Sys.time(), "%Y%m%d_%H%M"))
  mlflow_set_tag("mlflow.runName", run_display_name)  
  mlflow_set_tag("month_tag", month_tag)
  mlflow_set_tag("model", "EPD_GLMNET")
  mlflow_set_tag("version", "EPDModelling_Local")
  
  #-------------------
  # Data Source MetaData
  #--------------------
  mlflow_log_param("data_source", "Local")
  mlflow_log_param("container", "Local")
  mlflow_log_param("month_tag", month_tag)
  # In production, use multiple datasets that bind together to form development sample
  clean_df_1 <- readRDS(file.path(local_folder, "Datasets/clean_df_1_Aug2026.RDS"))
  clean_df_2 <- readRDS(file.path(local_folder, "Datasets/clean_df_2_Aug2026.RDS"))
  clean_df_3 <- readRDS(file.path(local_folder, "Datasets/clean_df_3_Aug2026.RDS"))
  
  # OOT Datasets. clean_df_4 is pre Dev Sample. 
  # clean_df_5 is post Dev Sample that we want to predict
  
  clean_df_4 <- readRDS(file.path(local_folder, "Datasets/clean_df_4_Aug2026.RDS"))
  clean_df_5 <- readRDS(file.path(local_folder, "Datasets/clean_df_5_Aug2026.RDS"))
  
  Complete_Sample <- bind_rows(clean_df_1, clean_df_2, clean_df_3, clean_df_4, clean_df_5) %>%
    mutate(
      epd = factor(epd),
      jointIndicator = factor(jointIndicator),
      NEW_OR_USED_FLAG = factor(NEW_OR_USED_FLAG),
      Program_Type = factor(Program_Type),
      FAMILY_CODE = factor(FAMILY_CODE),
      overallThinFile = factor(overallThinFile),
      IsFraud = factor(IsFraud)
    )
  
  # Dev Sample (clean_df_1,clean_df_2,clean_df_3)
  Dev_Sample <- Complete_Sample %>%
    filter(appno %in% clean_df_1$appno | appno %in% clean_df_2$appno | appno %in% clean_df_3$appno ) 
  
  mlflow_log_metric("dev_sample_row", nrow(Dev_Sample))
  mlflow_log_metric("dev_sample_col", ncol(Dev_Sample))
  
  # Split Train and Test
  split_seed = 502
  set.seed(split_seed)
  Train_split <- initial_split(Dev_Sample, prop = 0.7, strata = epd)
  
  epd_train <- training(Train_split) %>%
    arrange(FUNDING_DATE) 
  epd_test <- testing(Train_split) %>%
    arrange(FUNDING_DATE) 
  
  mlflow_log_param("split_seed", split_seed)
  mlflow_log_metric("train_n", nrow(epd_train))
  mlflow_log_metric("test_n", nrow(epd_test))
  
  mlflow_log_csv(as.data.frame(table(epd_train$epd)), file.path(tempdir(), "train_epd_table.csv"))
  mlflow_log_csv(as.data.frame(table(epd_test$epd)),  file.path(tempdir(), "test_epd_table.csv"))
  
  # ---- Variable Grouping ----
  # epd_application_vars are core application level features. These do not require WoE binning
  # All of these variables exist in the raw data
  epd_application_vars <- c(
    "appno","acctno","epd",
    "FUNDING_DATE","origapr","PTI","LTV","jointIndicator",
    "Program_Type","FAMILY_CODE",
    "stdpmt","DOWN_PAYMENT_PERCENTAGE","annualIncomeCombined","NEW_OR_USED_FLAG",
    'terms','ORIGINATION_SOURCE','overallThinFile',"TOTAL_FINANCED_AMOUNT",'NetTrades_Decision',
    "NumOfExceptions","openAccountsCount","IsFraud"
  )
  
  # epd_cat_vars are variables that that need one hot encoding.
  # Some of these variables are in the raw data, others are engineered in the recipe (Zero_Down_Flag) 
  epd_cat_vars <- c(
    "jointIndicator","NEW_OR_USED_FLAG",
    "Program_Type","FAMILY_CODE","ORIGINATION_SOURCE",
    "overallThinFile",
    "Zero_Down_Flag",
    "Extended_Term_Flag","Exceptions_1orMore","OpenAccts","IsFraud"
  )
  
  # epd_WoE_vars are bureau level variables requiring WoE encoding
  # All of these variables exist in the raw data
  epd_WoE_vars <- c(
    "ALL0439","ALL1300","ALL2428","ALL6230","ALL7116","AUA8220","ALL7938","IQF9510","ALL8151","AUA1300",
    "BCX7110","AUA6280","BCX1300","BCC6280","BCX5420","ILN7430","ILN1380","AUT0300","ILN7110","IQA9510","IQT9421",
    "IQT9425","MTA7430","IQT9426","PIL8220","ALL0000","ALL1361","ALL6250","ALL7170","ALL8725","ALX3510","AUA1380",
    "ILN6160","IQT9423","IQT9420",
    "BCC5520","MTA8220","AUA8320","IQT9417","ALL0448","BCX3423","ILN6200","BCC5627",
    "ALL8370","ALL2358","ALL7937","PIL8120","ALL8152",
    "BureauScore_application","NextGenScore_application"
  )
  
  epd_model_vars <- c(epd_application_vars, epd_WoE_vars)
  
  epd_train <- epd_train %>% dplyr::select(dplyr::all_of(epd_model_vars))
  epd_test  <- epd_test  %>% dplyr::select(dplyr::all_of(epd_model_vars))
  
  mlflow_log_metric("WoE_vars_used", length(epd_WoE_vars))
  mlflow_log_metric("application_vars_used", length(epd_application_vars))
  
  # -----------------------
  # Recipes
  # -----------------------
  # This is a two-recipe design to keep the modeling pipeline clean.
  # Recipe 1 handles all of the feature engineering and WoE binning. This ensures
  # that bureau variables are transformed consistently across train/test/OOT
  # Recipe 2 handles dummy encoding and the final model.
  
  # -----------------------
  # Recipe 1: Feature Engineering + WoE Binning
  # -----------------------
  
  epd_rec_bins <-
    recipe(epd ~ ., data = epd_train) %>%
    # Roles
    update_role(appno, new_role = "Application ID") %>%
    update_role(acctno, new_role = "Account ID") %>%
    update_role(FUNDING_DATE, new_role = "FUNDING DATE") %>%
    
    # ---- Feature engineering 
    # Data cleaning of extreme values
    
    step_mutate(
      annualIncomeCombined = if_else(annualIncomeCombined > 300000, 300000, annualIncomeCombined),
      origapr = if_else(origapr > 35, 35, origapr),
      stdpmt  = if_else(stdpmt > 3000, 3000, stdpmt),
      PTI     = if_else(PTI > 0.5, 0.5, PTI),
      LTV     = if_else(LTV > 1.7, 1.7, LTV),
      DOWN_PAYMENT_PERCENTAGE = if_else(DOWN_PAYMENT_PERCENTAGE > 80, 80, DOWN_PAYMENT_PERCENTAGE),
      DOWN_PAYMENT_PERCENTAGE = if_else(is.na(DOWN_PAYMENT_PERCENTAGE), 10, DOWN_PAYMENT_PERCENTAGE),
      terms = if_else(terms > 96, 96,terms),
      terms = if_else(is.na(terms), 84, terms),
      
      ORIGINATION_SOURCE = as.factor(ORIGINATION_SOURCE),
      Exceptions_1orMore = as.factor(case_when(NumOfExceptions >= 1 ~ "AtLeast1",
                                               TRUE ~ "NoExceptions")),
      OpenAccts = as.factor(if_else(openAccountsCount >= 1,"YesOpenAccts","NoOpenAccts")),
      Zero_Down_Flag = as.factor(if_else(DOWN_PAYMENT_PERCENTAGE == 0, "ZeroDown", "SomeDown")),
      Extended_Term_Flag = as.factor(if_else(terms >= 84,"84M+","LT84Months"))
    ) %>%
    # -- Log of Variables that are skewed
    step_log(annualIncomeCombined,offset = 1,base=exp(1)) %>%
    step_log(NetTrades_Decision,offset = 1, base=exp(1)) %>%
    # -- Remove Variables Not Needed In Modelling 
    step_rm(any_of(c('TOTAL_FINANCED_AMOUNT'))) %>%
    step_rm(any_of(c('CASH_DOWN_PAYMENT'))) %>%
    step_rm(any_of(c('NumOfExceptions'))) %>%
    step_rm(any_of(c('openAccountsCount'))) %>%
    
    
    # ---- Raw score cleanup ----
  step_mutate_at(contains("NextGenScore"), fn = ~ replace(.,. == 0, 650),id = "Replace_0") %>% #Replace 0 NG Score with 650
    step_mutate_at(contains("BureauScore"),fn = ~ ifelse(. %in% c(0) | . >= 9000, 625, .)) %>% #Replace 0 FICO Score with 625
    # -- Impute Medians for missing data
    step_impute_median(any_of(c(
      "origapr",
      "PTI",
      "LTV",
      "annualIncomeCombined",
      "stdpmt",
      "BureauScore_application",
      "NextGenScore_application",
      "DOWN_PAYMENT_PERCENTAGE",
      "terms",
      "NetTrades_Decision"
    ))) %>%
    
    # Monotonic Binning
    step_monobin_woe(
      all_of(epd_WoE_vars),
      outcome = epd,
      woe.gap = 0.1,
      sc.method = "separately"
    )
  
  # -----------------------
  # Prep Bins. - Freeze monotonic bins + WoE values and apply everywhere
  # -----------------------
  rec_bins_prep <- prep(epd_rec_bins, training = epd_train)
  
  idx <- which(vapply(rec_bins_prep$steps, inherits, logical(1), "step_monobin_woe"))
  
  if (length(idx) == 0) {
    stop("monobin_woe step not found in recipe after prep")
  }
  if (length(idx) > 1) {
    stop("Multiple monobin_woe steps found — unexpected")
  }
  
  bin_step <- rec_bins_prep$steps[[idx]]
  
  frozen_bins        <- bin_step$bins
  frozen_breaks_list <- bin_step$breaks_list
  frozen_columns     <- bin_step$columns
  
  mlflow_log_rds(frozen_bins, "woe_bins.rds")
  mlflow_log_rds(frozen_breaks_list, "woe_breaks.rds")  
  
  # -----------------------
  # Transform Training and Test Using Frozen Bins
  # -----------------------
  # Bake is used to apply transformations to epd_train_woe and epd_test_woe
  epd_train_woe <- bake(rec_bins_prep, new_data = epd_train)
  epd_test_woe  <- bake(rec_bins_prep, new_data = epd_test)
  
  # ============================================================
  # FEATURE REDUNDANCY REDUCTION (WoE Clustering)
  # ============================================================
  # Cluster WoE varibles by correlation and select one representative per cluster
  
  # ---- Identify WoE columns ----
  woe_cols <- names(epd_train_woe)[grepl("_woe$", names(epd_train_woe))]
  
  # ---- Build correlation matrix ----
  X_woe <- epd_train_woe %>%
    dplyr::select(dplyr::all_of(woe_cols)) %>%
    as.data.frame()
  
  corr_mat <- stats::cor(X_woe, use = "pairwise.complete.obs")
  
  # ---- Hierarchical clustering ----
  dist_mat <- stats::as.dist(1 - abs(corr_mat))
  hc <- stats::hclust(dist_mat, method = "average")
  
  # ---- Define clustering threshold ----
  corr_cutoff <- 0.80   # typical range: 0.80–0.90 
  clusters <- stats::cutree(hc, h = 1 - corr_cutoff)
  
  cluster_tbl <- tibble::tibble(
    variable = names(clusters),
    cluster  = as.integer(clusters)
  )
  
  # ---- Compute univariate AUC per variable ----
  y01 <- as.numeric(epd_train_woe$epd) - 1
  
  auc_tbl <- purrr::map_dfr(woe_cols, function(v) {
    df_tmp <- tibble::tibble(
      y = factor(y01),
      x = epd_train_woe[[v]]
    )
    
    auc_val <- tryCatch(
      yardstick::roc_auc_vec(truth = df_tmp$y, estimate = df_tmp$x, event_level = "second"),
      error = function(e) NA_real_
    )
    
    tibble::tibble(variable = v, auc = as.numeric(auc_val))
  })  
  
  # ---- Select best variable per cluster ----
  cluster_selected <- cluster_tbl %>%
    dplyr::left_join(auc_tbl, by = "variable") %>%
    dplyr::group_by(cluster) %>%
    dplyr::arrange(dplyr::desc(auc), .by_group = TRUE) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup()
  
  
  # ---- Drop weak pure noise clusters by AUC threshold ----
  auc_threshold <- 0.505
  
  cluster_selected <- cluster_selected %>%
    dplyr::filter(auc >= auc_threshold)
  
  woe_keep <- cluster_selected$variable
  woe_drop <- setdiff(woe_cols, woe_keep)
  
  # ---- Apply reduction to train/test ----
  epd_train_woe <- epd_train_woe %>%
    dplyr::select(-dplyr::all_of(woe_drop))
  
  epd_test_woe <- epd_test_woe %>%
    dplyr::select(-dplyr::all_of(woe_drop))
  
  # ---- Log results to MLflow ----
  mlflow_log_csv(cluster_tbl, file.path(tempdir(), "woe_clusters.csv"))
  mlflow_log_csv(cluster_selected, file.path(tempdir(), "woe_cluster_selected.csv"))
  
  dropped_tbl <- tibble::tibble(variable = woe_drop)
  mlflow_log_csv(dropped_tbl, file.path(tempdir(), "woe_dropped_variables.csv"))
  
  mlflow_log_metric("woe_variables_before", length(woe_cols))
  mlflow_log_metric("woe_variables_after", length(woe_keep))
  mlflow_log_metric("woe_auc_threshold", auc_threshold)
  
  message("WoE reduction complete: kept ", length(woe_keep),
          " variables, dropped ", length(woe_drop))                        
  
  
  # -----------------------
  # Recipe 2: Build Final Model using output from Recipe 1
  # -----------------------
  epd_rec_model <- recipe(epd ~ ., data = epd_train_woe) %>%
    # Roles
    update_role(appno, new_role = "Application ID") %>%
    update_role(acctno, new_role = "Account ID") %>%
    update_role(FUNDING_DATE, new_role = "FUNDING DATE") %>%
    # ---- Categorical handling ----
  # step_unknown → handles NA categories
  # step_other   → collapses rare levels (<5%) to reduce noise
  # step_novel   → handles unseen categories in future data
  step_unknown(all_of(epd_cat_vars), new_level = "missing") %>%
    step_other(all_of(epd_cat_vars), threshold = 0.05, other = "other") %>%
    step_novel(all_of(epd_cat_vars), new_level = "new") %>%
    # - Calculate the FUNDING_DATE quarter, year
    step_mutate(funding_qtr = paste0("Q", quarter(FUNDING_DATE))) %>%
    step_string2factor(funding_qtr, ordered = FALSE) %>%
    step_unknown(funding_qtr, new_level = "NA") %>%
    step_mutate(funding_yr = paste0("Y", year(FUNDING_DATE))) %>%
    step_string2factor(funding_yr, ordered = FALSE) %>%
    step_unknown(funding_yr, new_level = "NA") %>%
    
    # ---- One-hot encoding
    step_dummy(
      epd_cat_vars,
      funding_qtr,
      funding_yr,
      one_hot = FALSE # Returns all categories
    ) %>%
    # - Imputation & scaling (NON‑WoE numerics only) 
    step_normalize(any_of(c(
      "origapr",
      "PTI",
      "LTV",
      "annualIncomeCombined",
      "stdpmt",
      "DOWN_PAYMENT_PERCENTAGE",
      "terms",
      "NetTrades_Decision"
    ))) %>%
    # - Remove zero and near zero variance
    step_zv(all_predictors()) %>%
    step_nzv(all_predictors()) 
  
  
  # --- Function to apply frozen monotonic bins + WoE to new data ---
  apply_woe <- function(new_data, frozen_bins, frozen_columns, keep_original_cols = FALSE) {
    # Apply WoE using scorecard::woebin_ply
    woe_df <- scorecard::woebin_ply(
      dt = new_data %>% dplyr::select(dplyr::all_of(frozen_columns)),
      bins = frozen_bins,
      no_cores = 1,
      print_step = 0L
    )
    # Drop raw bureau variables unless requested
    if (!keep_original_cols) {
      new_data <- new_data %>% dplyr::select(-dplyr::all_of(frozen_columns))
    }
    # Return transformed dataset
    dplyr::bind_cols(new_data, woe_df)
  }
  
  
  # -----------------------
  # GLMNET Model Training
  # -----------------------
  # Fit an elastic net using the pre-processed and WoE variables
  # GLMNET provides the regularization between lasso and ridge
  # and allows for explainability for our leaders
  
  # Principal is to use cross-validation to select the ideal setup for the glmnet
  # penalty controls the strength of regularization. Large lambda is simpler model
  # mixture controls the balance between ridge and lasso.
  logistic_spec <- logistic_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    set_engine("glmnet")
  
  # Set Workflow
  wflow <- workflow() %>%
    add_model(logistic_spec) %>%
    add_recipe(epd_rec_model)
  
  # Use Period Fold for the cross validation
  period_fold <- sliding_period(
    data = epd_train_woe,
    index = FUNDING_DATE,
    period = "month",
    lookback = 12, # Each training fold uses previous 12 months of data
    assess_stop = 3, #Each validation fold uses 3 month window
    step     = 3, #Moves the window forward in 3 month increments
    complete = TRUE #Ensures each fold contains the full lookback + assessment window
  )
  
  set.seed(345)
  
  # Parallelization
  future::plan(sequential)
  
  # Tidymodels Control Object
  ctrl <- control_grid(
    allow_par = TRUE,
    verbose = FALSE,
    save_pred = TRUE
  )
  
  # Set Grid for Penalty and Mixture. Default is very wide for penalty so zoomed in quite a bit here
  penalty_vals <- 10^seq(-5, -1, length.out = 10) #Typically do length.out of 40
  mixture_vals <- c(0.0, 0.5, 1.0) #Typically span more mixture
  
  penalty_grid <- tidyr::crossing(
    penalty = penalty_vals,
    mixture = mixture_vals
  )
  
  #Tune Resampling to find best metrics
  tune_rs <- tune_grid(
    wflow,
    resamples = period_fold,
    grid = penalty_grid,
    metrics = metric_set(roc_auc),
    control = ctrl
  )
  
  tuning_metrics <- tune_rs %>% collect_metrics()
  mlflow_log_csv(tuning_metrics, file.path(tempdir(), "tuning_metrics.csv"))
  
  # Choose best Cross Validation
  best <- tune_rs %>%
    select_by_one_std_err(
      desc(penalty),
      desc(mixture),   # slight bias toward simpler models
      metric = "roc_auc"
    )
  
  best_penalty <- best$penalty[[1]]
  best_mixture <- best$mixture[[1]]
  
  best_mean <- tuning_metrics %>%
    dplyr::filter(.metric == "roc_auc", penalty == best_penalty, mixture == best_mixture) %>%
    dplyr::slice(1) %>%
    dplyr::pull(mean)
  
  
  mlflow_log_param("best_penalty", best_penalty)
  mlflow_log_param("best_mixture", best_mixture)
  mlflow_log_metric("best_cv_roc_auc_mean", best_mean)
  
  metrics_tbl <- tune_rs %>%
    collect_metrics() %>%
    dplyr::filter(.metric == "roc_auc")
  
  best_row <- metrics_tbl %>%
    dplyr::arrange(desc(mean)) %>%
    dplyr::slice(1)
  
  best_approx_sd <- best_row %>%
    dplyr::select(mean, std_err) %>%
    dplyr::mutate(
      approx_sd = std_err * sqrt(nrow(period_fold))
    ) %>%
    pull(approx_sd)
  
  mlflow_log_metric("approx_sd_Stability_GoalUnder_0.015", best_approx_sd)
  
  # -----------------------
  # Train model on full training sample with optimal mixture/penalty
  # -----------------------
  
  final_spec <- logistic_reg(
    penalty = best_penalty,
    mixture = best_mixture
  ) %>%
    set_engine("glmnet")
  
  final_fit <- wflow %>% update_model(final_spec) %>% 
    fit(epd_train_woe)
  
  # Calculate AUC on Train/Test and record
  
  train_prob <- predict(final_fit, new_data = epd_train_woe, type = "prob") %>%
    bind_cols(epd_train_woe %>% select(epd))
  train_auc <- roc_auc(train_prob, truth = epd, .pred_1, event_level = "second")$.estimate
  
  mlflow_log_metric("train_auc", train_auc)
  
  test_prob <- predict(final_fit, new_data = epd_test_woe, type = "prob") %>%
    bind_cols(epd_test_woe %>% select(epd))
  test_auc  <- roc_auc(test_prob,  truth = epd, .pred_1, event_level = "second")$.estimate
  
  mlflow_log_metric("test_auc",  test_auc)
  
  # Compute KS for train/test and record
  train_ks <- ks_from_roc(train_prob, epd, .pred_1)
  test_ks  <- ks_from_roc(test_prob,  epd, .pred_1)
  
  mlflow_log_metric("train_ks", train_ks)
  mlflow_log_metric("test_ks",  test_ks)
  
  # Log model RDS
  mlflow_log_rds(final_fit, file.path(tempdir(), paste0("final_fit_", month_tag, ".rds")))
  
  # Calculate Variable Importance and plot
  vi_tbl <- vip::vi(final_fit, lambda = best_penalty)
  mlflow_log_csv(vi_tbl, file.path(tempdir(), "variable_importance.csv"))
  
  glmnet_fit <- extract_fit_parsnip(final_fit)$fit
  
  vip_plot <- vip::vip(
    glmnet_fit,
    num_features = 20,
    lambda = best_penalty,
    geom = "col")
  
  vip_plot_gg <- ggplot(vip_plot, aes(
    x = reorder(Variable, Importance),
    y = Importance,
    fill = Sign
  )) +
    geom_col() +
    coord_flip() +
    labs(
      title = paste0("EPD GLMNET VIP (", month_tag, ")"),
      x = "Variable",
      y = "Importance"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(angle = 0, size = 9),
      legend.position = "bottom"
    )
  
  mlflow_log_plot_png(vip_plot_gg, file.path(tempdir(), "vip_top20.png"))
  
  vi_check <- vip::vi(final_fit, lambda = best_penalty)
  wrong_way <- vi_check %>%
    dplyr::filter(grepl("_woe$", Variable), (Sign == "NEG" & Importance > 0))
  
  mlflow_log_csv(wrong_way, file.path(tempdir(), "wrong_way.csv"))
  
  # -----------------------
  # Model Calibration (Platt Scaling)
  # -----------------------
  # Adjust predicted probabilities to better match observed outcomes.
  # Use Logistic calibration (GAM smoothing)
  
  # Score the train data
  pred_train <- predict(final_fit, epd_train_woe, type = "prob") %>%
    bind_cols(epd_train_woe %>% select(epd))
  
  # Convert outcome to numeric 0/1
  pred_train <- pred_train %>%
    mutate(epd_numeric = as.numeric(epd) - 1)
  
  # Create decile bins based on predicted probability of default (.pred_1)
  calibration_tbl_train <- pred_train %>%
    mutate(decile = ntile(.pred_1, 20)) %>%
    group_by(decile) %>%
    summarise(
      avg_pred = mean(.pred_1),
      obs_rate = mean(epd_numeric),
      count    = n(),
      .groups  = "drop"
    )
  
  mlflow_log_csv(calibration_tbl_train, file.path(tempdir(), "calibration_tbl_train.csv"))
  
  brier_train <- pred_train %>% brier_class(truth = epd,.pred_0) %>% select(.estimate) %>% as.numeric()
  mlflow_log_metric("brier_train", brier_train)
  
  # Calibration plot
  cal_plot_train <- ggplot(calibration_tbl_train, aes(x = avg_pred, y = obs_rate)) +
    geom_point(size = 3) +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
    labs(
      x     = "Average Predicted EPD Probability (per twentile)",
      y     = "Observed EPD Rate (per twentile)",
      title = "Calibration Plot: Predicted vs. Observed EPD: Training Set"
    ) +
    theme_minimal()
  
  mlflow_log_plot_png(cal_plot_train, file.path(tempdir(), "cal_plot_train.png"))
  
  # Score the test data
  pred_test <- predict(final_fit, epd_test_woe, type = "prob") %>%
    bind_cols(epd_test_woe %>% select(epd))
  
  # Convert outcome to numeric 0/1
  pred_test <- pred_test %>%
    mutate(epd_numeric = as.numeric(epd) - 1)
  
  # Create decile bins based on predicted probability of default (.pred_1)
  calibration_tbl_test <- pred_test %>%
    mutate(decile = ntile(.pred_1, 20)) %>%
    group_by(decile) %>%
    summarise(
      avg_pred = mean(.pred_1),
      obs_rate = mean(epd_numeric),
      count    = n(),
      .groups  = "drop"
    )
  
  mlflow_log_csv(calibration_tbl_test, file.path(tempdir(), "calibration_tbl_test.csv"))
  
  # Calibration plot
  cal_plot_test <- ggplot(calibration_tbl_test, aes(x = avg_pred, y = obs_rate)) +
    geom_point(size = 3) +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
    labs(
      x     = "Average Predicted EPD Probability (per twentile)",
      y     = "Observed EPD Rate (per twentile)",
      title = "Calibration Plot: Predicted vs. Observed EPD: Training Set"
    ) +
    theme_minimal()
  
  mlflow_log_plot_png(cal_plot_test, file.path(tempdir(), "cal_plot_test.png"))
  
  brier_test <- pred_test %>% brier_class(truth = epd,.pred_0) %>% select(.estimate) %>% as.numeric()
  mlflow_log_metric("brier_test", brier_test)
  
  # -----------------------
  # Register model in MLflow Model Registry (RDS workflow)
  # -----------------------
  
  run_id <- mlflow_get_run()$run_id
  
  model_rds_name <- paste0("final_fit_", month_tag, ".rds")
  model_name     <- paste0("EPD_GLMNET_", month_tag)
  
  
  # 1) Ensure registered model exists
  try(
    mlflow_create_registered_model(name = model_name),
    silent = TRUE
  )
  
  
  # 2) Create a new model version pointing at the RDS artifact
  mlflow_create_model_version(
    name   = model_name,
    source = paste0("runs:/", run_id, "/", model_rds_name),
    run_id = run_id
  )
  
  mlflow_set_tag("registered_model", model_name)
  mlflow_set_tag("model_artifact_type", "workflow_rds")
  
  # -----------------------
  #  Platt Scaling
  # -----------------------
  
  # Step 1: Generate predictions on the test set
  pred_valid <- predict(final_fit, epd_test_woe, type = "prob") %>%
    bind_cols(epd_test_woe %>% select(epd))
  
  # Step 2: Fit the logistic calibrator
  # smooth = TRUE (default) uses a GAM; smooth = FALSE uses simple logistic
  cal_logistic <- cal_estimate_logistic(
    pred_valid,
    truth    = epd,
    estimate = c(.pred_0,.pred_1),
    smooth   = TRUE
  )
  
  # -----------------------
  #  Get Actuals Results
  # -----------------------
  
  by_month_actuals <- Complete_Sample %>% 
    mutate(year_month = format(as.Date(FUNDING_DATE), "%Y-%m"),
           epd = as.numeric(epd)-1)%>%
    group_by(year_month) %>% summarise(sum = sum(epd,na.rm=TRUE), count = n(),percent = (sum/count)*100)
  
  actuals_out_csv <- file.path(local_folder, paste0("by_month_actuals_", month_tag, ".csv"))
  readr::write_csv(by_month_actuals, actuals_out_csv)
  mlflow_log_csv(by_month_actuals, file.path(tempdir(), "by_month_actuals.csv"))
  mlflow_set_tag("local_output_actual_csv", actuals_out_csv)
  
  # -----------------------
  #  Score Complete Sample
  # -----------------------
  
  Complete_Sample_woe <- bake(rec_bins_prep, new_data = Complete_Sample) %>%
    dplyr::select(-dplyr::all_of(woe_drop))
  
  
  scored <- predict(final_fit, new_data = Complete_Sample_woe, type = "prob") %>%
    bind_cols(Complete_Sample)   # keep raw columns for reporting
  
  
  # Apply Calibration
  
  scored_calibrated <- cal_apply(scored,cal_logistic) %>%
    rename(EPD_Probability = .pred_1)
  
  # -----------------------
  # Monte Carlo Forecast + Backtesting
  # -----------------------
  # PURPOSE:
  #   - Convert predictions into portfolio-level ranges
  #   - Evaluate whether intervals are well calibrated
  #
  # BACKTEST:
  #   - Excludes most recent 3 months (incomplete performance)
  #   - Checks % of actuals within predicted bands
  #
  # OUTPUT:
  #   - coverage_rate
  #   - calibration_flag (under / over / well)
  
  by_month_mc <- MonteCarlo_byMonth(
    scored_df = scored_calibrated,
    date_col  = "FUNDING_DATE",
    prob_col  = "EPD_Probability",
    n_sims    = 2000,
    seed      = 893
  )
  
  mc_out_csv <- file.path(local_folder, paste0("by_month_glmnet_monteCarloPreds_", month_tag, ".csv"))
  readr::write_csv(by_month_mc, mc_out_csv)
  mlflow_log_csv(by_month_mc, file.path(tempdir(), "by_month_monteCarlo.csv"))
  mlflow_set_tag("local_output_monteCarlo_csv", mc_out_csv)
  
  
  # -----------------------
  # Return last 3 month's riskiest loans
  # -----------------------
  risky_scores <- 
    scored_calibrated %>% 
    filter(appno %in% clean_df_5$appno) %>%
    arrange(desc(EPD_Probability)) %>%
    select(appno,acctno,EPD_Probability,FUNDING_DATE,NextGenScore_application,stdpmt,origapr,
           DOWN_PAYMENT_PERCENTAGE,LTV,PTI,
           NumOfExceptions,
           AUA1300,ALX3510,IQT9423,ALL8151
    ) %>%
    slice_head(n=100)
  
  risky_csv <- file.path(local_folder, paste0("risky_apps_", month_tag, ".csv"))
  readr::write_csv(risky_scores, risky_csv)
  mlflow_log_csv(risky_scores, file.path(tempdir(), "risky_apps.csv"))
  mlflow_set_tag("risky_apps_csv", risky_csv)
  
  # -----------------------
  # Backtest 
  # -----------------------
  
  Complete_Sample_minus_current <- Complete_Sample %>%
    filter(appno %in% clean_df_1$appno | appno %in% clean_df_2$appno | appno %in% clean_df_3$appno | appno %in% clean_df_4$appno ) 
  
  
  by_month_actuals_minus_current <- Complete_Sample_minus_current %>% 
    mutate(year_month = format(as.Date(FUNDING_DATE), "%Y-%m"),
           epd = as.numeric(epd)-1)%>%
    group_by(year_month) %>% summarise(sum = sum(epd,na.rm=TRUE), count = n(),percent = (sum/count)*100)
  
  bt_results <- Backtest_EPD_Intervals(
    by_month_actuals = by_month_actuals_minus_current,
    by_month_mc     = by_month_mc
  )
  
  # Log summary metrics
  mlflow_log_metric("coverage_rate", bt_results$summary$coverage_rate)
  mlflow_log_metric("coverage_expected", bt_results$summary$expected_coverage)
  
  # Log diagnostic flags
  mlflow_set_tag("coverage_status", bt_results$summary$calibration_flag)
  
  # Save detail tables
  mlflow_log_csv(bt_results$detail, file.path(tempdir(), "interval_backtest_detail.csv"))
  mlflow_log_csv(bt_results$miss_summary, file.path(tempdir(), "interval_backtest_miss_summary.csv"))
  mlflow_log_csv(bt_results$summary, file.path(tempdir(), "interval_backtest_summary.csv"))
  
  
})  



