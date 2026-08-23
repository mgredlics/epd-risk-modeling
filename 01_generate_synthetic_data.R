library(dplyr)
library(lubridate)

# ------------------------------------------------------------
# Logistic scaling helper
# ------------------------------------------------------------
scale_prob_to_rate <- function(raw_prob, target_rate) {
  
  # Clip raw probabilities to avoid 0 or 1
  eps <- 1e-6
  raw_prob <- pmin(pmax(raw_prob, eps), 1 - eps)
  
  logit_raw <- log(raw_prob / (1 - raw_prob))
  logit_target <- log(target_rate / (1 - target_rate))
  
  shift <- logit_target - mean(logit_raw)
  
  scaled_prob <- plogis(logit_raw + shift)
  
  # Clip again to ensure valid probabilities
  scaled_prob <- pmin(pmax(scaled_prob, eps), 1 - eps)
  
  return(scaled_prob)
}

# ------------------------------------------------------------
# Synthetic EPD generator risk signals
# ------------------------------------------------------------
generate_epd <- function(df, base_rate = 0.015) {
  
  df <- df %>%
    mutate(
      Zero_Down_Flag = if_else(DOWN_PAYMENT_PERCENTAGE == 0, "ZeroDown", "SomeDown"),
      funding_qtr = paste0("Q", quarter(FUNDING_DATE)),
      funding_yr  = year(FUNDING_DATE),
      IsException = if_else(NumOfExceptions > 0, "Yes", "No"),
      IsFraud = "No"
    )
  
  # Boosted signal weights to raise AUC to ~0.85
  risk_score <-
    1.8 * (df$Zero_Down_Flag == "ZeroDown") +
    0.65 * scale(df$origapr) +
    0.50 * scale(df$ALX3510) +
    0.50 * scale(df$IQT9423) +
    0.40 * scale(df$AUA1300) +
    0.15 * scale(df$stdpmt) -
    0.35 * scale(df$DOWN_PAYMENT_PERCENTAGE) -
    0.85 * scale(df$NextGenScore_application) +
    0.55 * scale(df$LTV) +
    0.50 * scale(df$PTI) +
    0.35 * scale(df$ALL8151) +
    0.35 * scale(df$ALL2428) +
    
    # Mild macro drift so Monte Carlo 90% bands cover actuals (~0.90 coverage)
    -0.15 * (df$funding_qtr == "Q1") +
    0.10 * (df$funding_qtr == "Q2") +
    0.05 * (df$funding_qtr == "Q3") +
    -0.10 * (df$funding_yr == 2022) +
    0.05 * (df$funding_yr == 2023) +
    0.15 * (df$funding_yr == 2024) +
    0.20 * (df$funding_yr == 2025)
  
  # Convert and scale
  raw_prob <- plogis(as.numeric(risk_score))
  scaled_prob <- scale_prob_to_rate(raw_prob, base_rate)
  
  # Draw outcomes
  df$epd <- factor(rbinom(nrow(df), 1, scaled_prob), levels = c(0,1))
  df
}

# ------------------------------------------------------------
# Synthetic dataset generator
# ------------------------------------------------------------
generate_synthetic_clean_df <- function(n_rows,
                                        start_appno,
                                        start_acctno,
                                        start_date = "2022-01-01",
                                        end_date   = "2025-12-31") {
  
  date_seq <- seq.Date(as.Date(start_date), as.Date(end_date), by = "day")
  
  # ---- Globally unique keys ----
  appno  <- seq(start_appno, start_appno + n_rows - 1)
  acctno <- seq(start_acctno, start_acctno + n_rows - 1)
  
  # ---- FUNDING_DATE ----
  FUNDING_DATE <- sample(date_seq, n_rows, replace = TRUE)
  
  # ---- ZeroDown generation (10%) ----
  zero_down_flag <- rbinom(n_rows, 1, 0.10)
  
  DOWN_PAYMENT_PERCENTAGE <- ifelse(
    zero_down_flag == 1,
    0,
    runif(n_rows, 1, 50)
  )
  
  # ---- Application numeric variables ----
  origapr                <- round(runif(n_rows, 6, 25), 2)
  PTI                    <- runif(n_rows, 0.02, 0.18)
  LTV                    <- runif(n_rows, 0.70, 1.30)
  annualIncomeCombined   <- round(runif(n_rows, 20000, 300000), 2)
  stdpmt                 <- round(runif(n_rows, 200, 1200), 2)
  terms                  <- sample(36:96, n_rows, replace = TRUE)
  NetTrades_Decision     <- runif(n_rows, 0, 85)
  TOTAL_FINANCED_AMOUNT  <- runif(n_rows, 5000, 50000)
  openAccountsCount      <- sample(c(0,0,0,1,2,3), n_rows, replace = TRUE)
  
  # ---- Application categorical variables ----
  jointIndicator       <- factor(sample(c("Yes","No"), n_rows, replace = TRUE))
  Program_Type         <- factor(sample(c("A","B","C","D"), n_rows, replace = TRUE))
  FAMILY_CODE          <- factor(sample(c("F1","F2","F3","F4"), n_rows, replace = TRUE))
  NEW_OR_USED_FLAG     <- factor(sample(c("New","Used"), n_rows, replace = TRUE))
  ORIGINATION_SOURCE   <- factor(sample(c("Online","Dealer"), n_rows, replace = TRUE))
  overallThinFile      <- factor(sample(c("Yes","No"), n_rows, replace = TRUE))
  
  # ---- Bureau variables (whole numbers only) ----
  bureau_vars <- list(
    ALL0439 = round(rnorm(n_rows, 50, 20)),
    ALL1300 = round(runif(n_rows, 0, 100)),
    ALL2428 = round(rpois(n_rows, 20)),
    ALL6230 = round(rnorm(n_rows, 40, 15)),
    ALL7116 = round(rnorm(n_rows, 60, 25)),
    AUA8220 = round(rnorm(n_rows, 55, 18)),
    ALL7938 = round(rnorm(n_rows, 45, 22)),
    IQF9510 = round(rnorm(n_rows, 600, 50)),
    ALL8151 = round(rnorm(n_rows, 30, 10)),
    AUA1300 = round(runif(n_rows, 0, 100)),
    BCX7110 = round(rnorm(n_rows, 70, 30)),
    AUA6280 = round(rnorm(n_rows, 50, 20)),
    BCX1300 = round(runif(n_rows, 0, 100)),
    BCC6280 = round(rnorm(n_rows, 45, 15)),
    BCX5420 = round(rnorm(n_rows, 55, 18)),
    ILN7430 = round(rnorm(n_rows, 40, 12)),
    ILN1380 = round(rnorm(n_rows, 35, 10)),
    AUT0300 = round(rnorm(n_rows, 50, 20)),
    ILN7110 = round(rnorm(n_rows, 60, 25)),
    IQA9510 = round(rnorm(n_rows, 650, 40)),
    IQT9421 = round(rnorm(n_rows, 620, 45)),
    IQT9425 = round(rnorm(n_rows, 610, 50)),
    MTA7430 = round(rnorm(n_rows, 55, 18)),
    IQT9426 = round(rnorm(n_rows, 600, 55)),
    PIL8220 = round(rnorm(n_rows, 48, 15)),
    ALL0000 = round(rnorm(n_rows, 30, 10)),
    ALL1361 = round(rnorm(n_rows, 40, 12)),
    ALL6250 = round(rnorm(n_rows, 50, 20)),
    ALL7170 = round(rnorm(n_rows, 60, 25)),
    ALL8725 = round(rnorm(n_rows, 45, 15)),
    ALX3510 = round(rnorm(n_rows, 55, 18)),
    AUA1380 = round(rnorm(n_rows, 35, 10)),
    ILN6160 = round(rnorm(n_rows, 40, 12)),
    IQT9423 = round(rnorm(n_rows, 620, 45)),
    IQT9420 = round(rnorm(n_rows, 600, 50)),
    BCC5520 = round(rnorm(n_rows, 48, 15)),
    MTA8220 = round(rnorm(n_rows, 55, 18)),
    AUA8320 = round(rnorm(n_rows, 45, 15)),
    IQT9417 = round(rnorm(n_rows, 610, 50)),
    ALL0448 = round(rnorm(n_rows, 35, 10)),
    BCX3423 = round(rnorm(n_rows, 50, 20)),
    ILN6200 = round(rnorm(n_rows, 40, 12)),
    BCC5627 = round(rnorm(n_rows, 48, 15)),
    ALL8370 = round(rnorm(n_rows, 45, 15)),
    ALL2358 = round(rnorm(n_rows, 40, 12)),
    ALL7937 = round(rnorm(n_rows, 45, 15)),
    PIL8120 = round(rnorm(n_rows, 48, 15)),
    ALL8152 = round(rnorm(n_rows, 30, 10)),
    BureauScore_application = round(runif(n_rows, 350, 850)),
    NextGenScore_application = round(runif(n_rows, 450, 850))
  )
  
  df <- data.frame(
    appno, acctno, FUNDING_DATE,
    origapr, PTI, LTV, annualIncomeCombined, stdpmt,
    DOWN_PAYMENT_PERCENTAGE, terms, NetTrades_Decision,
    TOTAL_FINANCED_AMOUNT, openAccountsCount,
    jointIndicator, Program_Type, FAMILY_CODE, NEW_OR_USED_FLAG,
    ORIGINATION_SOURCE, overallThinFile,
    NumOfExceptions = sample(c(0,0,0,1,2,3), n_rows, replace = TRUE)
  )
  
  df <- cbind(df, bureau_vars)
  
  df <- generate_epd(df)
  
  return(df)
}

# ------------------------------------------------------------
# Generate and save all five datasets
# ------------------------------------------------------------
output_folder <- "C:/Users/mgred/OneDrive/Documents/Data Science/EarlyDefault/Datasets/"
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

saveRDS(
  generate_synthetic_clean_df(
    100000, 1, 1000000,
    start_date = "2023-05-01",
    end_date   = "2024-04-30"
  ),
  file.path(output_folder, "clean_df_1_Aug2026.RDS")
)

saveRDS(
  generate_synthetic_clean_df(
    100000, 1000001, 1100000,
    start_date = "2024-05-01",
    end_date   = "2025-04-30"
  ),
  file.path(output_folder, "clean_df_2_Aug2026.RDS")
)

saveRDS(
  generate_synthetic_clean_df(
    100000, 2000001, 1200000,
    start_date = "2025-05-01",
    end_date   = "2026-04-30"
  ),
  file.path(output_folder, "clean_df_3_Aug2026.RDS")
)

saveRDS(
  generate_synthetic_clean_df(
    130000, 3000001, 1300000,
    start_date = "2022-01-01",
    end_date   = "2023-04-30"
  ),
  file.path(output_folder, "clean_df_4_Aug2026.RDS")
)

saveRDS(
  generate_synthetic_clean_df(
    25000, 3020001, 1320000,
    start_date = "2026-05-01",
    end_date   = "2026-07-31"
  ),
  file.path(output_folder, "clean_df_5_Aug2026.RDS")
)
