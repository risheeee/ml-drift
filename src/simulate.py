import numpy as np
import pandas as pd
from pathlib import Path
import logging
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/creditcard.csv")
WINDOWS_PATH = Path("data/windows")
N_WINDOWS = 6
WINDOW_LABELS = {
    1: "baseline_1", 2: "baseline_2", 3: "baseline_3",
    4: "covariate_drift", 5: "concept_drift", 6: "catastrophic_drift",
}

def load_and_split() -> list[pd.DataFrame]:
    df = pd.read_csv(RAW_PATH).sort_values("Time").reset_index(drop=True)
    windows = np.array_split(df, N_WINDOWS)
    logger.info(f"Loaded {len(df)} rows, split into {N_WINDOWS} windows of ~{len(windows[0])} rows each")
    return windows

def inject_covariate_drift(df: pd.DataFrame, rotation_angle: float = 0.3, seed: int = 37) -> pd.DataFrame:
    """
    mean changed, variance changed, but orthogonality is preserved. hence relationship to target stays the same. this is the most basic level of drift as only the feature distributon changes.
    """

    df = df.copy()
    features = ["V1", "V2", "V3", "V4"]
    X = df[features].values
    np.random.seed(seed)
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.eye(4)
    R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c     # 'givens' rotation
    X_rotated = X @ R.T     # matrix multi
    df[features] = X_rotated

    # log-shifting amount distribution to simulate change in spending patterns.
    amount_log = np.log1p(df["Amount"])
    df["Amount"] = np.expm1(amount_log + amount_log.std())
    logger.info(f"Window 4: Covariate drift injected. V1 mean: {X[:,0].mean():.3f} -> {X_rotated[:,0].mean():.3f}")
    return df

def inject_concept_drift(df: pd.DataFrame, noise_scale: float = 1.5, seed: int = 37) -> pd.DataFrame:
    """
    here we flip the fraud signal. i.e, if fraud used to be high amount, it becomes low. hence the model gets confused. concept drift. impact is serious as the feature - label relatuiionship changes now.
    """

    df = df.copy()
    np.random.seed(seed)
    fraud_mask = df["Class"] == 1
    mediaan_amount = df["Amount"].median()
    df.loc[fraud_mask, "Amount"] = np.maximum(0, 2 * mediaan_amount - df.loc[fraud_mask, "Amount"])
    n_fraud = fraud_mask.sum()
    for feature in ["V1", "V2", "V3"]:
        df.loc[fraud_mask, feature] += np.random.normal(0, noise_scale, n_fraud)
    logger.info(f"Window 5: Concept drift injected on {n_fraud} fraud rows.")
    return df

def inject_catastrophic_drift(df: pd.DataFrame, fraud_multiplier: int = 5, seed: int = 37) -> pd.DataFrame:
    """
    here we simulate a system failure as we destroy key features. we also change class distribution, i.e, fraud gets oversampled.
    """

    df = df.copy()
    np.random.seed(seed)
    df["V1"] = np.random.normal(0, 1, len(df))
    df["V14"] = 0.0
    df["V17"] = 0.0
    fraud_rows = df[df["Class"] == 1]
    extra_fraud = fraud_rows.sample(n=len(fraud_rows) * fraud_multiplier, replace=True, random_state=seed)
    df = pd.concat([df, extra_fraud], ignore_index=True).sample(frac=1, random_state=seed)
    logger.info(f"Window 6: Catastrophic drift. Fraud rate: {df['Class'].mean():.3%}")
    return df

def create_windows():
    WINDOWS_PATH.mkdir(parents=True, exist_ok=True)
    windows = load_and_split()
    injectors = {4: inject_covariate_drift, 5: inject_concept_drift, 6: inject_catastrophic_drift}
    for i, window_df in enumerate(windows, start=1):
        out_dir = WINDOWS_PATH / f"window_{i}"
        out_dir.mkdir(exist_ok=True)
        if i in injectors:
            window_df = injectors[i](window_df)
        out_path = out_dir / "data.parquet"
        window_df.to_parquet(out_path, index=False)
        logger.info(f"Window {i} ({WINDOW_LABELS[i]}): {len(window_df)} rows -> {out_path}")
    reference = pd.read_parquet(WINDOWS_PATH / "window_1" / "data.parquet")
    reference.to_parquet(WINDOWS_PATH / "reference.parquet", index=False)
    logger.info("Reference distribution saved (Window 1)")
 
 
if __name__ == "__main__":
    create_windows()