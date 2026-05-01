import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, roc_curve
import optuna
import mlflow
import mlflow.sklearn
import joblib
import logging
import yaml
from datetime import datetime
 
import features as feat_module
 
logger = logging.getLogger(__name__)

WINDOWS_PATH = Path("data/windows")
MODEL_DIR = Path("models")
CONFIG_PATH = Path("configs/config.yaml")

def load_config() -> dict:
    with open(CONFIG_PATH) as f: return yaml.safe_load(f)

def load_training_data(up_to_window: int) -> pd.DataFrame:
    """Point-in-time correct: only use windows that existed before the trigger."""
    dfs = []
    for i in range(1, up_to_window + 1):
        path = WINDOWS_PATH / f"window_{i}" / "data.parquet"
        if path.exists(): dfs.append(pd.read_parquet(path))
    if not dfs: raise FileNotFoundError(f"No windows found up to window {up_to_window}")
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows from windows 1-{up_to_window}")
    return df

def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> float:
    fpr, tpr = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, tpr)
    return float(tpr[min(idx, len(tpr) - 1)])

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "auprc": average_precision_score(y_true, y_score),
        "auroc": roc_auc_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division = 0),
        "recall_at_5pct_fpr": recall_at_fpr(y_true, y_score, 0.05),
    }

def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """optuna for maximizing auprc"""

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
    }
    skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 37)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = GradientBoostingClassifier(**params, random_state = 37)
        model.fit(X[train_idx], y[train_idx])
        scores.append(average_precision_score(y[val_idx], model.predict_proba(X[val_idx])[:, 1]))
    return np.mean(scores)

def train(up_to_window: int = 3, run_hpo: bool = True, n_trials: int = 30, trigger_reason: str = "initial_training") -> str:
    config = load_config()
    MODEL_DIR.mkdir(parents = True, exist_ok = True)
    mlflow.set_experiment("fraud-drift-pipeline")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("trigger_reason", trigger_reason)
        mlflow.set_tag("trained_on_windows", f"1-{up_to_window}")
        mlflow.set_tag("timestamp", datetime.utcnow().isoformat())

        df = load_training_data(up_to_window)
        pipeline = feat_module.fit_and_save(df)
        X, y = feat_module.transform(df, pipeline)

        logger.info(f"Training set: {X.shape}, fraud rate: {y.mean():.4%}")
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("fraud_rate", round(float(y.mean()), 6))
        mlflow.log_param("trained_on_windows", up_to_window)

        if run_hpo:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda t: objective(t, X, y), n_trials=n_trials)
            best_params = study.best_params
            logger.info(f"Best HPO: {best_params}, AUPRC: {study.best_value:.4f}")
        else:
            best_params = config["model"]["default_params"]

        mlflow.log_params(best_params)
        model = GradientBoostingClassifier(**best_params, random_state = 37)
        model.fit(X, y)

        holdout_df = df.iloc[int(len(df) * 0.8):]
        X_hold, y_hold = feat_module.transform(holdout_df, pipeline)
        metrics = compute_metrics(y_hold, model.predict_proba(X_hold)[:, 1])
        mlflow.log_metrics(metrics)
        logger.info(f"Holdout metrics: {metrics}")
 
        model_path = MODEL_DIR / f"model_{run_id}.joblib"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        mlflow.register_model(f"runs:/{run_id}/model", "fraud-detector")
 
        logger.info(f"Training complete. Run ID: {run_id}")
        return run_id
    
if __name__ == "__main__":
    train(up_to_window = 3, run_hpo = False)