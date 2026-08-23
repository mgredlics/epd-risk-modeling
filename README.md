# Early Payment Default (EPD) Forecasting Engine

An end-to-end credit risk modeling pipeline built in R using `tidymodels` and `glmnet`, tracked with `MLflow`. This project estimates account-level Early Payment Default probabilities and generates portfolio-level monthly confidence intervals via Monte Carlo simulation.

> **Note on Data:** All data in this repository is programmatically synthesized.
---

## Architecture Overview

1. **Synthetic Data Generation:** Generates multi-vintage application and bureau attributes with class imbalance and volatility.
2. **Feature Engineering (Two-Recipe Architecture):**
   - *Recipe 1:* Custom monotonic binning and Weight of Evidence (WoE) transformation for bureau predictors. Parameters are learned strictly on training partitions to prevent leakage.
   - *Feature Redundancy Reduction:* Hierarchical correlation clustering paired with univariate AUC pruning.
   - *Recipe 2:* Dummy encoding, normalization of non-WoE continuous drivers, and near-zero-variance filtering.
3. **Modeling & Validation:**
   - Elastic Net Logistic Regression (`glmnet`) tuned over a rolling temporal cross-validation window (`sliding_period`: 12-month train, 3-month validate).
   - Probability calibration via GAM-smoothed Platt Scaling.
4. **Operations Forecasting:**
   - 2,000-iteration Monte Carlo simulation producing monthly confidence bands.

---

## Model Performance

| Metric | Training Split | Testing Split | Target Benchmark |
| :--- | :--- | :--- | :--- |
| **ROC AUC** | 0.88 | 0.88 | ~0.85 |
| **KS Statistic** | 0.60 | 0.59 | > 0.50 |
| **Brier Score** | 0.039 | 0.039 | Well-Calibrated |
| **Interval Coverage Rate** | — | 82.7% | 80.0% Expected |

---

## How to Run

1. **Generate Data:**
   ```R
   source("01_generate_synthetic_data.R")
   ```
2. **Start Local MLflow Server:**
   ```powershell
   python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri "sqlite:///C:/mlflow_local/mlflow.db" --default-artifact-root "file:///C:/mlflow_local/artifacts"
   ```
3. **Execute Pipeline:**
   ```R
   source("02_epd_model_pipeline.R")
   ```
