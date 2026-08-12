# Project Walkthrough - Medical Clinical Diagnostic Pipeline with Leakage Prevention

This document details the successful completion of the secure medical diagnostics pipeline project. We have built an interactive machine learning pipeline application that prevents **Data Leakage** during medical clinical data preparation, using **FastAPI** for the backend API and **Streamlit** for the frontend dashboard.

---

## 🛠️ Implemented Components

The project consists of the following components implemented in the workspace:

1. **Pipeline Utilities ([pipeline_utils.py](file:///c:/ibm%20prOJECT/pipeline_utils.py))**:
   - Implements a secure Scikit-Learn `Pipeline` wrapping a `ColumnTransformer` (handling missing value imputation, scaling, and categorical encoding) and an estimator (Random Forest, Gradient Boosting, or Logistic Regression).
   - Generates an artificial 10% missing value rate in continuous fields (`serum_sodium`, `platelets`) to simulate raw clinical records.
   - Provides functions for **Secure Cross-Validation** (nested pipeline preprocessing) and **Leaky Cross-Validation** (global preprocessing before splitting), demonstrating the difference in performance metrics.
   - Computes model generalization comparisons on clean holdout test sets split before any data manipulation.
   - Traces patient metrics step-by-step through imputation, scaling, and encoding stages.

2. **FastAPI Backend Service ([app_api.py](file:///c:/ibm%20prOJECT/app_api.py))**:
   - Exposes `/data-info` for dataset summaries, types, and missing metrics.
   - Exposes `/train` to accept configurations, perform training on secure vs. leaky setups, and return comparative evaluation metrics, confusion matrices, ROC coordinates, and feature importances.
   - Exposes `/predict` to evaluate individual patient diagnostics through the trained secure pipeline and trace intermediate states.
   - Implements robust Pydantic schemas (nullable continuous variables) to handle missing patient records.

3. **Streamlit Interface ([app_dashboard.py](file:///c:/ibm%20prOJECT/app_dashboard.py))**:
   - Designed a premium dark-themed clinical dashboard using custom CSS styles.
   - Features sidebar configurations for classifier, imputation strategy, scaling, cross-validation splits, and test set fractions.
   - Implements four interactive tabs:
     1. **💡 Data Leakage & Security**: Educational charts and textual comparisons explaining why leaky preprocessing fails in production.
     2. **📊 Dataset Statistics**: Visual summaries, target distribution pie charts, and missing value indicators.
     3. **📈 Model Comparison**: Side-by-side metric tables, comparison bar charts, ROC curves, and confusion matrices showing the validation inflation gap.
     4. **🩺 Patient Risk Calculator**: Live risk estimator form with sliders, supporting missing values, and tracing the pipeline step-by-step.

---

## 🔍 Verification Details

We successfully executed and verified both background services:
- **FastAPI backend** running on [http://127.0.0.1:8000](http://127.0.0.1:8000).
- **Streamlit frontend** running on [http://127.0.0.1:8501](http://127.0.0.1:8501).

### Test Case Execution
To test the pipeline's robustness, the browser subagent configured a patient profile with missing values (simulating incomplete clinical logs):
- **Ejection Fraction**: `25%`
- **Serum Creatinine**: `2.5 mg/dL`
- **Serum Sodium**: `[Missing Value]` (NaN)
- **Platelet Count**: `[Missing Value]` (NaN)
- **All other features**: Kept at clinical defaults.

**Result**:
- The API ran successfully, returning a **46.0% mortality probability** (classified as **MODERATE RISK**).
- The pipeline trace verified that:
  - `serum_sodium` was securely imputed to `136.5` (the training set mean).
  - `platelets` was securely imputed to `264,012.2` (the training set mean).
  - Continuous values were successfully scaled (StandardScaler z-scores computed from train statistics: Ejection Fraction = `-1.077`, Creatinine = `1.020`).
  - Prediction was generated without any information leakage from the test profile.

---

## 🖼️ Visual Demonstrations

### Patient Diagnostics Results Screen
The screenshot below shows the calculations and the secure preprocessing execution trace:

![Patient Diagnostic Result](C:/Users/rohit/.gemini/antigravity-ide/brain/e4e34a9f-d46a-4350-90ac-1bc82c2173b5/prediction_results_1786368818786.png)

### Video Walkthrough of User Flow
Below is a video recording capturing the navigation, pipeline configuration, and diagnostic evaluation:

![SafeClinical Verification Flow](C:/Users/rohit/.gemini/antigravity-ide/brain/e4e34a9f-d46a-4350-90ac-1bc82c2173b5/safeclinical_working_flow_1786368775943.webp)
