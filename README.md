# Telco Customer Churn Prediction (Business-Oriented Machine Learning)

## Overview
This project predicts whether a telecom customer is likely to churn using the Kaggle Telco Customer Churn dataset (7,043 customers, 21 features).  
My main goal was not only to build a model with good statistical performance, but to make the output usable for real decision-making. In churn prediction, the “best” model depends on what the business values more: catching as many churners as possible, or avoiding unnecessary retention actions. That’s why I focused heavily on evaluation, probability thresholds, and cost-based optimisation.

### Data Source
https://www.kaggle.com/datasets/ramasub78/telco-customer-churn-prediction
---

## Dataset
- Rows: 7,043 customers  
- Target: `Churn` (Yes/No)  
- Mix of numeric and categorical features  
- Class imbalance: churn is the minority class (around 26%)

This imbalance matters because many common metrics (especially accuracy) can look acceptable even when the model is not good at identifying churners.

---

## Step-by-step approach

### 1) Data loading and initial checks
I started by loading the CSV and reviewing:
- the shape of the dataset
- data types
- missing values
- the churn distribution (to confirm imbalance)

This step is important because it reveals data quality issues early and prevents errors later in the pipeline.

---

### 2) Data cleaning (key issue: `TotalCharges`)
The Telco dataset has a known issue: `TotalCharges` is stored as text and includes blank strings in some rows.  
If not fixed, this breaks numeric transformations (like median imputation).

What I did:
- stripped whitespace from object columns to remove hidden blanks
- converted `TotalCharges` to numeric using `errors="coerce"` (blank values become NaN)

This ensures the numeric pipeline can impute missing values safely and consistently.

---

### 3) Train/test split (leakage-safe)
Before doing any scaling or encoding, I split the data into train and test sets using stratification.

Why this matters:
- preprocessing steps must be learned only from training data
- stratification keeps the churn ratio similar in train and test, which makes evaluation more reliable

---

### 4) Preprocessing pipeline (production-style)
Instead of manually encoding and scaling outside the model, I built a reusable preprocessing pipeline using `ColumnTransformer`.

Numeric features:
- median imputation
- standard scaling (helps logistic regression)

Categorical features:
- most-frequent imputation
- one-hot encoding with `handle_unknown="ignore"`

Why this is “production-style”:
- all preprocessing is applied consistently in training and inference
- it reduces the risk of accidental leakage
- it makes the workflow reproducible and easy to deploy

---

### 5) Model comparison (cross-validation)
I compared multiple models using 5-fold cross-validation:
- Logistic Regression (balanced class weights)
- Random Forest (balanced)
- XGBoost

I evaluated them using:
- ROC-AUC (ranking ability overall)
- PR-AUC (more informative for imbalanced churn)
- Precision, Recall, F1 (to understand tradeoffs)

Key finding:
Logistic Regression performed best overall on this dataset, especially in recall and PR-AUC, which are critical for churn detection.

---

### 6) Why PR-AUC was important
Because churn is the minority class, PR-AUC gives a clearer picture than ROC-AUC in many churn scenarios.

ROC-AUC can look good even if the model produces too many false alarms or struggles to identify churners cleanly.  
PR-AUC focuses on what matters most here: precision-recall performance for the churn class.

---

### 7) Threshold tuning and cost-based optimisation
By default, classifiers use a threshold of 0.50 (probability >= 0.5 means “churn”).  
In churn prediction, this default is rarely optimal.

Instead, I implemented a cost-based threshold optimisation process:
- I generated predictions as probabilities
- I evaluated multiple thresholds
- for each threshold, I computed the confusion matrix (TP, FP, FN, TN)
- I selected the threshold that best fit a retention cost scenario

This approach is closer to how churn models are used in real businesses: the goal is not only to be “accurate,” but to make good decisions under budget and cost constraints.

Best threshold found: **0.25**

This threshold prioritised catching churners:
- Recall (Churn) = 0.94  
- Precision (Churn) = 0.41  
- Accuracy = 0.62  

Interpretation:
Lowering the threshold increased sensitivity. The model catches most churners, but it also flags more non-churners. This is often acceptable if retention actions are relatively cheap (e.g., emails, small offers) and missing churners is costly.

---

## Final model decision
Final model: Logistic Regression (class-weight balanced)  
Final operating point: threshold = 0.25 (selected via cost-based optimisation)

This solution is intentionally decision-focused: it shows how the same model can behave very differently depending on the chosen threshold and business constraints.

---

## Tech stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost (used for model comparison)

---

## What this project demonstrates
- Clean, leakage-safe pipeline design
- Strong understanding of class imbalance and metric selection
- Cross-validation-based model comparison
- Decision threshold tuning based on business costs
- Ability to explain results in business terms, not just ML metrics

---

## How to run
1. Open the notebook
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost
