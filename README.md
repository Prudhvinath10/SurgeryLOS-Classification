# 🏥 Surgery Length of Stay Prediction

**Leveraging Machine Learning to Predict Post-Surgery Length of Stay**

This repository contains our final project for the UT ORIE Summer 2024 Applied Projects. Our goal? Predict how long a patient will stay in the hospital after surgery, based on data available either **before** or **after** surgery. This could help hospitals assign inpatient/outpatient status more intelligently—without relying on a stethoscope and a dream.

---

## 📌 Objective

Develop robust ML models to predict **Post-Surgery Length of Stay (LOS)** into 4 discrete categories using structured surgical data. Our mission: optimize hospital capacity by forecasting how long patients will hang out in recovery, from **pre-op data** to **post-op insights**.

---

## 🧠 What We Did

- 🧽 **Outlier Detection** with flexible IQR filtering (1.0–3.0)
- 🤕 **Missing Value Imputation**:
  - *Numerical*: Median by groups like Age Bucket, Sex, Service
  - *Categorical*: Mode fill, 'Unknown' handling
- 📊 **Feature Engineering & Dropping** of redundant or sparsely populated features
- 📦 **One-Hot Encoding** to convert categorical features into 6000+ columns of pure regret
- 🧮 **Model Training**:
  - Random Forest Classifier (RFC)
  - XGBoost (XGB)
- 🧪 **Model Evaluation**: Accuracy, Recall, Precision, F1-Score, Cross-Validation
- 🔍 **Feature Selection**: Forward & Backward Selection to reduce dimensionality
- 💡 **Explainability**: SHAP analysis to see what features actually mattered

---

## 🔬 Results Summary

### Post-Operative Predictions

| Model        | Accuracy | F1 Score | Notes |
|--------------|----------|----------|-------|
| RFC (Base)   | 93.43%   | 93%      | Strong baseline |
| RFC (Tuned)  | 95.24%   | 95%      | With feature selection |
| XGB (Tuned)  | 83.60%   | 84%      | Improved with deeper trees |

### Pre-Operative Predictions

| Model        | Accuracy | F1 Score | Notes |
|--------------|----------|----------|-------|
| RFC (Base)   | 89.39%   | 89%      | Decent from early data |
| RFC (Tuned)  | 93.29%   | 93%      | With feature selection |
| XGB (Tuned)  | 78.64%   | 79%      | Trying its best |

---

## 🧬 Data Overview

- Total records: 139,634
- Final cleaned post-op dataset: 38,487 rows
- Final cleaned pre-op dataset: 30,409 rows
- Features after one-hot encoding: 5,000–6,000+
- Target variable: `LOS_4_Groups`

| Class | Label           |
|-------|-----------------|
| 0     | 60+ hours       |
| 1     | <12 hours       |
| 2     | 12–36 hours     |
| 3     | 36–60 hours     |

---

## 📓 Notebooks

| File | Description |
|------|-------------|
| `DataCleaning_Graphs.ipynb` | Outlier analysis, missing value imputation, and visualizations of numerical features |
| `RFC.ipynb` | Random Forest Classifier model training and evaluation (Pre-Op and Post-Op); includes feature selection and cross-validation |
| `XGB.ipynb` | XGBoost Classifier with Bayesian optimization; includes SHAP analysis and final model performance evaluation |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- `pandas`, `numpy`, `scikit-learn`, `xgboost`
- `matplotlib`, `seaborn`, `shap`
- `bayesian-optimization`
- Jupyter Notebooks

---

## 💡 Future Work

- Replace one-hot encoding with **ClinicalBERT** embeddings
- Explore deep learning models or regression for precise LOS predictions
- Investigate SHAP insights further for clinical interpretability
- Analyze misclassification patterns across patient subgroups
- Avoid working with 6,000-column CSVs ever again

---

## 📁 Repository Structure

```text
surgiLOS/
├── data/                      # Raw and processed data (excluded from repo)
│   ├── raw/
│   └── processed/
├── notebooks/                 # All Jupyter notebooks for the project
│   ├── DataCleaning_Graphs.ipynb
│   ├── RFC.ipynb
│   └── XGB.ipynb
├── src/                       # Python scripts for cleaning, modeling, utils (optional)
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── tuning.py
│   └── utils.py
├── models/                    # Saved model files (.pkl)
├── reports/                   # Final report and presentation PDFs
│   ├── Final_Report.pdf
│   └── Final_Presentation.pdf
├── requirements.txt           # Python dependencies
├── README.md                  # This beautiful monstrosity
└── .gitignore                 # Ignoring sensitive/temporary files
