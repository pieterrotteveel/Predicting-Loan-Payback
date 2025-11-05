# Loan Payback Prediction

End-to-end notebook for training gradient-boosted models (LightGBM & XGBoost) to predict whether a loan will be paid back. Includes lightweight EDA, feature engineering, cross-validated training, and submission file generation.

---

## Repository layout

```
.
├── prediction_models_for_loan_payback.ipynb   # Main notebook
├── train.csv                                  # Training data
├── test.csv                                   # Test data
├── sample_submission.csv                      # Kaggle/competition template
├── lgbm_submission.csv                        # LightGBM predictions (created by the notebook)
├── xgb_submission.csv                         # XGBoost predictions (created by the notebook)
└── README.md
```

---

## Data

Columns (train):

* **Numerical:** `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`, `loan_paid_back` *(target)*
* **Categorical:** `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`

Basic checks performed:

* Drop `id`
* Verify shapes (train: 593,994 × 12; test: 254,569 × 11)
* No nulls/duplicates detected
* Moderate class imbalance (~80% repaid)

---

## Environment & setup

Tested with Python ≥ 3.10.

### Quickstart

```bash
# 1) (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) install dependencies
pip install numpy pandas scikit-learn lightgbm xgboost scipy matplotlib seaborn

# 3) run the notebook
jupyter lab  # or jupyter notebook
```

---

## What the notebook does

### 1) Imports & configuration

Common scientific Python stack plus LightGBM/XGBoost. Warnings suppressed for cleaner logs.

### 2) Load data

Reads `train.csv`, `test.csv`, `sample_submission.csv`, and the two submission files (if present).

### 3) EDA (spot checks)

* Summary statistics for numeric features
* Correlation heatmap
* Distribution plots & boxplots
* Skewness computation

### 4) Preprocessing

* **Target removal** from numeric set before plotting correlations
* **Skew handling:** `log1p` on highly skewed numeric features (`annual_income`, `debt_to_income_ratio`)
* **Outlier clipping:** IQR clipping per numeric column (train & test aligned)
* **Shift checks:** train vs. test KDE overlays for key features

### 5) Feature engineering

Adds domain-inspired features to enrich signal:

* Ratios / capacities:

  * `loan_to_income = loan_amount / (annual_income + 1)`
  * `total_debt = debt_to_income_ratio * annual_income`
  * `available_income = annual_income * (1 - debt_to_income_ratio)`
  * `affordability = available_income / (loan_amount + 1)`
  * `payment_to_income = monthly_payment / (annual_income/12 + 1)`
* Payment proxy:

  * `monthly_payment = loan_amount * (1 + interest_rate/100) / 12`
* Composite risk and interactions:

  * `risk_score = 40*dti + 30*(1 - credit_score/850) + 2*interest_rate`
  * `credit_interest = credit_score * interest_rate / 100`
  * `income_credit = log1p(annual_income) * credit_score / 1000`
  * `debt_loan = debt_to_income_ratio * log1p(loan_amount)`
* Log transforms:

  * `log_income = log1p(annual_income)`
  * `log_loan   = log1p(loan_amount)`

### 6) Encoding

`LabelEncoder` for all categorical columns; fitted on train and applied to test (ensures alignment).

### 7) Modeling

**Cross-validation:** StratifiedKFold (5 folds) using ROC-AUC.

* **LightGBM (GBDT):** tuned parameters (e.g., `n_estimators=1320`, `num_leaves=93`, `max_depth=5`, `learning_rate=0.05`, subsampling/regularization).

  * Reported fold AUCs: `[0.9235, 0.9239, 0.9220, 0.9230, 0.9221]`
  * **OOF ROC-AUC:** **0.92291**

* **XGBoost (hist):** tuned parameters (`max_depth=6`, `n_estimators=732`, `learning_rate≈0.0669`, regularization, `max_bin=504`, etc.).

  * 5-fold CV performed; per-fold AUCs are printed in the notebook logs.

### 8) Training full models & inference

* Fit final LightGBM on the full train set
* Generate test probabilities for both models
* **Save submissions:**

  * `lgbm_submission.csv` with `loan_paid_back` probabilities from LightGBM
  * `xgb_submission.csv` with `loan_paid_back` probabilities from XGBoost

> Both files follow the `sample_submission.csv` schema: `id,loan_paid_back`.

---

## How to reproduce

1. Place `train.csv`, `test.csv`, and `sample_submission.csv` in the repo root.
2. Open `prediction_models_for_loan_payback.ipynb`.
3. Run all cells top-to-bottom.

   * CV metrics print to the console.
   * `lgbm_submission.csv` and `xgb_submission.csv` are written to disk.

---

## Notes & tips

* **Imbalance:** The target is ~80/20; ROC-AUC is the primary metric to avoid accuracy pitfalls.
* **Categoricals:** `grade_subgrade` is informative; label encoding is simple—consider target or CatBoost encoding for potential gains.
* **Reproducibility:** `random_state=42` is used across models/splits.
* **Performance:** On the provided settings, LightGBM achieves ~0.923 OOF ROC-AUC.

---

## Possible improvements (next steps)

* Calibrated probabilities (Platt/Isotonic) and threshold tuning for business KPIs
* Target/cat encoding for high-cardinality categories
* Feature interaction search (e.g., polynomial on select ratios)
* SHAP analysis for interpretability & bias checks
* Model ensembling / stacking (blend LGBM & XGB)
* Hyperparameter search with Optuna
* Robust pipelines with `sklearn` `ColumnTransformer` + `Pipeline`
* Train/serving parity and model versioning

---

## License

Specify a license here (e.g., MIT). If omitted, the project is “all rights reserved” by default.

---
