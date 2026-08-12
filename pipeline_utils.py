import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Define numerical and categorical columns
NUM_COLS = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
CAT_COLS = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
TARGET_COL = 'DEATH_EVENT'

def load_data(filepath="Heart_failure_clinical_records_dataset.csv", missing_pct=0.10, seed=42):
    """
    Loads the clinical records dataset and artificially injects missing values (NaN)
    in specific numerical columns to simulate raw clinical data requiring imputation.
    """
    df = pd.read_csv(filepath)
    df_clean = df.copy() # Save a pristine copy
    
    # Inject missing values (NaN) in 'serum_sodium' and 'platelets' to show imputation logic
    np.random.seed(seed)
    for col in ['serum_sodium', 'platelets']:
        mask = np.random.rand(len(df)) < missing_pct
        df.loc[mask, col] = np.nan
        
    return df, df_clean

def get_preprocessor(imputer_strategy='mean', scaling_method='standard'):
    """
    Returns a ColumnTransformer containing numerical and categorical preprocessors.
    """
    # Numerical Preprocessing: Imputation -> Scaling
    if scaling_method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputer_strategy)),
        ('scaler', scaler)
    ])
    
    # Categorical Preprocessing: Imputation (mode) -> OneHotEncoding
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUM_COLS),
            ('cat', cat_transformer, CAT_COLS)
        ]
    )
    return preprocessor

def get_estimator(estimator_type='rf', seed=42):
    """
    Returns the classifier model based on selection.
    """
    if estimator_type == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    elif estimator_type == 'gb':
        return GradientBoostingClassifier(random_state=seed)
    elif estimator_type == 'lr':
        return LogisticRegression(max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")

def run_secure_cv(X, y, preprocessor, estimator, cv_folds=5, seed=42):
    """
    Runs cross-validation SECURELY. Preprocessing is nested inside the pipeline.
    Imputer fits and Scaler fits happen ONLY on the training fold of each split.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
    
    # Average the metrics
    metrics = {
        'accuracy': float(np.mean(cv_results['test_accuracy'])),
        'precision': float(np.mean(cv_results['test_precision'])),
        'recall': float(np.mean(cv_results['test_recall'])),
        'f1': float(np.mean(cv_results['test_f1'])),
        'roc_auc': float(np.mean(cv_results['test_roc_auc']))
    }
    return metrics

def run_leaky_cv(X, y, preprocessor, estimator, cv_folds=5, seed=42):
    """
    Runs cross-validation WITH LEAKAGE. 
    Preprocessing (fit_transform) is applied globally on the entire dataset first.
    Then CV is run on the preprocessed features. This leaks validation/test fold statistics
    (means, standard deviations, imputation values) into the training folds.
    """
    # Fit preprocessor globally
    X_preprocessed = preprocessor.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(estimator, X_preprocessed, y, cv=cv, scoring=scoring, return_train_score=False)
    
    # Average the metrics
    metrics = {
        'accuracy': float(np.mean(cv_results['test_accuracy'])),
        'precision': float(np.mean(cv_results['test_precision'])),
        'recall': float(np.mean(cv_results['test_recall'])),
        'f1': float(np.mean(cv_results['test_f1'])),
        'roc_auc': float(np.mean(cv_results['test_roc_auc']))
    }
    return metrics

def run_holdout_comparison(df, imputer_strategy='mean', scaling_method='standard', estimator_type='rf', test_size=0.2, seed=42):
    """
    Demonstrates the generalization gap on a clean holdout test set split *before* any operations.
    - Secure Model: Prep fitted ONLY on train. True evaluation on holdout test.
    - Leaky Model: Prep fitted globally on train+test. Model evaluated on preprocessed test.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Clean train-test split before preprocessing
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    # ------------------
    # SECURE PIPELINE FLOW
    # ------------------
    preprocessor_sec = get_preprocessor(imputer_strategy, scaling_method)
    estimator_sec = get_estimator(estimator_type, seed)
    
    pipeline_sec = Pipeline(steps=[
        ('preprocessor', preprocessor_sec),
        ('classifier', estimator_sec)
    ])
    
    # Fit only on train
    pipeline_sec.fit(X_train_raw, y_train)
    
    # Predict and evaluate on unseen holdout test
    y_pred_sec = pipeline_sec.predict(X_test_raw)
    y_prob_sec = pipeline_sec.predict_proba(X_test_raw)[:, 1]
    
    # Calculate feature importances from secure pipeline
    classifier = pipeline_sec.named_steps['classifier']
    preprocessor = pipeline_sec.named_steps['preprocessor']
    
    # Retrieve feature names from OneHotEncoder and numerical features
    num_features = NUM_COLS
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = list(cat_encoder.get_feature_names_out(CAT_COLS))
    except Exception:
        cat_features = CAT_COLS
    feature_names = num_features + cat_features
    
    importances = []
    if hasattr(classifier, 'feature_importances_'):
        importances = list(classifier.feature_importances_)
    elif hasattr(classifier, 'coef_'):
        importances = list(np.abs(classifier.coef_[0]))
        
    feat_imp = [{"feature": name, "importance": float(imp)} for name, imp in zip(feature_names, importances)]
    feat_imp = sorted(feat_imp, key=lambda x: x['importance'], reverse=True)
    
    secure_holdout_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_sec)),
        'precision': float(precision_score(y_test, y_pred_sec, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_sec, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_sec, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_sec))
    }
    
    # ------------------
    # LEAKY PIPELINE FLOW
    # ------------------
    preprocessor_leak = get_preprocessor(imputer_strategy, scaling_method)
    estimator_leak = get_estimator(estimator_type, seed)
    
    # Preprocess globally (Train + Test combined)
    X_global_preprocessed = preprocessor_leak.fit_transform(X)
    
    # Recover train/test split on globally preprocessed data
    X_train_leak = X_global_preprocessed[X_train_raw.index]
    X_test_leak = X_global_preprocessed[X_test_raw.index]
    
    # Fit model on leaky train
    estimator_leak.fit(X_train_leak, y_train)
    
    # Predict and evaluate on leaky test
    y_pred_leak = estimator_leak.predict(X_test_leak)
    y_prob_leak = estimator_leak.predict_proba(X_test_leak)[:, 1]
    
    leaky_holdout_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_leak)),
        'precision': float(precision_score(y_test, y_pred_leak, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_leak, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_leak, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_leak))
    }
    
    # Cross Validation comparison
    preprocessor_cv_sec = get_preprocessor(imputer_strategy, scaling_method)
    estimator_cv_sec = get_estimator(estimator_type, seed)
    secure_cv_metrics = run_secure_cv(X, y, preprocessor_cv_sec, estimator_cv_sec, cv_folds=5, seed=seed)
    
    preprocessor_cv_leak = get_preprocessor(imputer_strategy, scaling_method)
    estimator_cv_leak = get_estimator(estimator_type, seed)
    leaky_cv_metrics = run_leaky_cv(X, y, preprocessor_cv_leak, estimator_cv_leak, cv_folds=5, seed=seed)
    
    # Generate ROC curve coordinates for plotting
    fpr_sec, tpr_sec, _ = roc_curve(y_test, y_prob_sec)
    fpr_leak, tpr_leak, _ = roc_curve(y_test, y_prob_leak)
    
    # Generate Confusion Matrix
    cm_sec = confusion_matrix(y_test, y_pred_sec).tolist()
    cm_leak = confusion_matrix(y_test, y_pred_leak).tolist()
    
    return {
        'secure_cv': secure_cv_metrics,
        'leaky_cv': leaky_cv_metrics,
        'secure_holdout': secure_holdout_metrics,
        'leaky_holdout': leaky_holdout_metrics,
        'roc_secure': {'fpr': fpr_sec.tolist(), 'tpr': tpr_sec.tolist(), 'auc': secure_holdout_metrics['roc_auc']},
        'roc_leaky': {'fpr': fpr_leak.tolist(), 'tpr': tpr_leak.tolist(), 'auc': leaky_holdout_metrics['roc_auc']},
        'cm_secure': cm_sec,
        'cm_leaky': cm_leak,
        'feature_importances': feat_imp
    }

def single_patient_diagnostic(patient_data, pipeline_fitted):
    """
    Evaluates a single patient's diagnostics through the fitted secure pipeline.
    Traces raw inputs -> preprocessed values -> final probability predictions.
    """
    # Create DataFrame from patient input
    df_patient = pd.DataFrame([patient_data])
    
    # Extract components of pipeline
    preprocessor = pipeline_fitted.named_steps['preprocessor']
    classifier = pipeline_fitted.named_steps['classifier']
    
    # Preprocess patient data and trace dimensions/shapes
    preprocessed_data = preprocessor.transform(df_patient)
    
    # Predict probabilities
    prob = float(pipeline_fitted.predict_proba(df_patient)[0, 1])
    prediction = int(pipeline_fitted.predict(df_patient)[0])
    
    # Detail the transformation trace
    num_imputed = preprocessor.named_transformers_['num'].named_steps['imputer'].transform(df_patient[NUM_COLS])
    num_scaled = preprocessor.named_transformers_['num'].named_steps['scaler'].transform(num_imputed)
    
    cat_imputed = preprocessor.named_transformers_['cat'].named_steps['imputer'].transform(df_patient[CAT_COLS])
    cat_encoded = preprocessor.named_transformers_['cat'].named_steps['onehot'].transform(cat_imputed)
    
    trace = {
        'raw_features': patient_data,
        'numerical_imputed': dict(zip(NUM_COLS, num_imputed[0].tolist())),
        'numerical_scaled': dict(zip(NUM_COLS, num_scaled[0].tolist())),
        'categorical_imputed': dict(zip(CAT_COLS, cat_imputed[0].tolist())),
        'categorical_encoded_shape': cat_encoded.shape,
        'preprocessed_array': preprocessed_data[0].tolist(),
        'mortality_probability': prob,
        'prediction': prediction
    }
    return trace
