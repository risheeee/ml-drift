import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging
 
logger = logging.getLogger(__name__)
 
MODEL_DIR = Path("models")
PIPELINE_PATH = MODEL_DIR / "feature_pipeline.joblib"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
TARGET = "Class"

class AmountTransformer(BaseEstimator, TransformerMixin):
    """
    log normalize amount (currenlty heavily right skwed)
    """
    def fit(self, X, y = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Amount_log"] = np.log1p(X["Amount"])
        return X.drop(columns = ["Amount", "Time"], errors = "ignore")
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names: list[str] = []

    def fit(self, X, y = None):
        self.feature_names = PCA_FEATURES + ["Amount_log"]
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features at inference time: {missing}. Possible upstream schema drift.")
        return X[self.feature_names_].values
    
    def get_feature_names(self) -> list[str]:
        return self.feature_names
    
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("amount", AmountTransformer()),
        ("select", FeatureSelector()),
        ("scale", StandardScaler()),
    ])

def fit_and_save(train_df: pd.DataFrame) -> Pipeline:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline()
    pipeline.fit(train_df.drop(columns=[TARGET]))
    joblib.dump(pipeline, PIPELINE_PATH)
    logger.info(f"Feature pipeline saved to {PIPELINE_PATH}")
    return pipeline
 
 
def load_pipeline() -> Pipeline:
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"No feature pipeline at {PIPELINE_PATH}. Run training first.")
    return joblib.load(PIPELINE_PATH)
 
 
def transform(df: pd.DataFrame, pipeline: Pipeline | None = None) -> tuple[np.ndarray, np.ndarray]:
    if pipeline is None: pipeline = load_pipeline()
    y = df[TARGET].values
    X = pipeline.transform(df.drop(columns=[TARGET]))
    return X, y
 
 
def get_feature_names() -> list[str]:
    return load_pipeline().named_steps["select"].get_feature_names_out()