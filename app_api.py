from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import pipeline_utils

app = FastAPI(title="Medical Clinical Diagnostic Pipeline API", version="1.0.0")

# Global reference to cache the trained secure pipeline
FITTED_PIPELINE = None

class TrainConfig(BaseModel):
    estimator_type: str = Field(default="rf", description="Model type: 'rf' (Random Forest), 'gb' (Gradient Boosting), 'lr' (Logistic Regression)")
    imputer_strategy: str = Field(default="mean", description="Imputation strategy: 'mean', 'median', 'most_frequent'")
    scaling_method: str = Field(default="standard", description="Scaling method: 'standard' (StandardScaler), 'minmax' (MinMaxScaler)")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of cross-validation folds")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Holdout test set fraction")

class PatientData(BaseModel):
    age: float | None = Field(default=None, description="Age in years")
    anaemia: int = Field(..., description="0 = No, 1 = Yes (presence of anaemia)")
    creatinine_phosphokinase: float | None = Field(default=None, description="Level of CPK enzyme in the blood (mcg/L)")
    diabetes: int = Field(..., description="0 = No, 1 = Yes (presence of diabetes)")
    ejection_fraction: float | None = Field(default=None, description="Percentage of blood leaving the heart at each contraction")
    high_blood_pressure: int = Field(..., description="0 = No, 1 = Yes (presence of hypertension)")
    platelets: float | None = Field(default=None, description="Platelets in the blood (kiloplatelets/mL)")
    serum_creatinine: float | None = Field(default=None, description="Level of serum creatinine in the blood (mg/dL)")
    serum_sodium: float | None = Field(default=None, description="Level of serum sodium in the blood (mEq/L)")
    sex: int = Field(..., description="0 = Female, 1 = Male")
    smoking: int = Field(..., description="0 = Non-smoker, 1 = Smoker")
    time: float | None = Field(default=None, description="Follow-up period in days")


@app.get("/data-info")
def get_data_info():
    """
    Returns information about the loaded heart failure dataset, including
    dimensions, class distributions, and count of missing features.
    """
    try:
        df, df_clean = pipeline_utils.load_data()
        
        # Dimensions
        rows, cols = df.shape
        
        # Missing values (after injection)
        missing_counts = df.isnull().sum().to_dict()
        
        # Data types and list of columns
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        
        # Class distribution of target
        target_counts = df[pipeline_utils.TARGET_COL].value_counts().to_dict()
        total_targets = sum(target_counts.values())
        target_dist = {str(k): {"count": int(v), "percentage": float(v / total_targets)} for k, v in target_counts.items()}
        
        # Summary statistics
        stats = df.describe().to_dict()
        
        return {
            "total_records": rows,
            "columns": list(df.columns),
            "data_types": dtypes,
            "missing_values": missing_counts,
            "class_distribution": target_dist,
            "summary_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

@app.post("/train")
def train_pipelines(config: TrainConfig):
    """
    Trains secure and leaky pipelines under the specified configuration.
    Returns comparison metrics, confusion matrices, ROC coordinates,
    and caches the secure pipeline for single-patient predictions.
    """
    global FITTED_PIPELINE
    try:
        df, _ = pipeline_utils.load_data()
        
        # Run comparison and fetch all results
        results = pipeline_utils.run_holdout_comparison(
            df=df,
            imputer_strategy=config.imputer_strategy,
            scaling_method=config.scaling_method,
            estimator_type=config.estimator_type,
            test_size=config.test_size,
            seed=42
        )
        
        # Train and cache the global secure pipeline on the training split
        # We perform a split and train a pipeline on the train portion of holdout comparison
        X = df.drop(columns=[pipeline_utils.TARGET_COL])
        y = df[pipeline_utils.TARGET_COL]
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=config.test_size, stratify=y, random_state=42
        )
        
        preprocessor = pipeline_utils.get_preprocessor(config.imputer_strategy, config.scaling_method)
        estimator = pipeline_utils.get_estimator(config.estimator_type, 42)
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', estimator)
        ])
        pipeline.fit(X_train, y_train)
        
        # Update the global fitted pipeline reference
        FITTED_PIPELINE = pipeline
        
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
def predict_patient(patient: PatientData):
    """
    Predicts mortality risk for a patient using the cached secure pipeline.
    If no model has been trained, it fits a default model first.
    """
    global FITTED_PIPELINE
    try:
        # Load default pipeline if none is trained yet
        if FITTED_PIPELINE is None:
            df, _ = pipeline_utils.load_data()
            X = df.drop(columns=[pipeline_utils.TARGET_COL])
            y = df[pipeline_utils.TARGET_COL]
            
            # Default to RF, mean imputer, standard scaler
            preprocessor = pipeline_utils.get_preprocessor('mean', 'standard')
            estimator = pipeline_utils.get_estimator('rf', 42)
            
            FITTED_PIPELINE = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', estimator)
            ])
            # Fit on standard 80% train split
            from sklearn.model_selection import train_test_split
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            FITTED_PIPELINE.fit(X_train, y_train)
            
        patient_dict = patient.model_dump()
        trace = pipeline_utils.single_patient_diagnostic(patient_dict, FITTED_PIPELINE)
        return trace
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction trace failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
