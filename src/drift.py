import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sqlite3
import logging
import yaml
from datetime import datetime, UTC
 
logger = logging.getLogger(__name__)
 
CONFIG_PATH = Path("configs/config.yaml")
DB_PATH = Path("data/drift_store.db")
WINDOWS_PATH = Path("data/windows")
MONITOR_FEATURES = ["V1", "V2", "V3", "V4", "V14", "V17", "Amount"]
 
 
def load_config() -> dict:
    with open(CONFIG_PATH) as f: return yaml.safe_load(f)

def init_db():
    DB_PATH.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS drift_metrics (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 window_id INTEGER, feature TEXT, metric_type TEXT,
                 value REAL, threshold REAL, triggered INTEGER, 
                 computed_at TEXT)
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS model_performance(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 window_id INTEGER, auprc REAL, auroc REAL,
                 recall_at_5pct_fpr REAL, score_dist_psi REAL,
                 computer_at TEXT)
    """)
    conn.commit()
    conn.close()

def write_drift_metrics(window_id: int, results: dict):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(UTC)
    rows = [(window_id, f, mt, v, th, int(tr), now)
            for f, metrics in results.items()
            for mt, (v, th, tr) in metrics.items()]
    conn.executemany(
        "INSERT INTO drift_metrics (window_id,feature,metric_type,value,threshold,triggered,computed_at) VALUES (?,?,?,?,?,?,?)",
        rows)
    conn.commit(); conn.close()

def write_performance_metrics(window_id: int, metrics: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO model_performance (window_id,auprc,auroc,recall_at_5pct_fpr,score_dist_psi,computed_at) VALUES (?,?,?,?,?,?)",
        (window_id, metrics.get("auprc"), metrics.get("auroc"),
         metrics.get("recall_at_5pct_fpr"), metrics.get("score_dist_psi"),
         datetime.now(UTC)))
    conn.commit(); conn.close()

def read_drift_history() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM drift_metrics ORDER BY computed_at", conn)
    conn.close(); return df
 
def read_performance_history() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM model_performance ORDER BY computed_at", conn)
    conn.close(); return df

# Population Stability Index (PSI) test - https://medium.com/model-monitoring-psi/population-stability-index-psi-ab133b0a5d42

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI < 0.10: stable | 0.10-0.20: monitor | > 0.20: retrain
    """

    breakpoints = np.unique(np.nanpercentile(reference, np.linspace(0, 100, n_bins + 1)))
    eps = 1e-6
    ref_pct = np.histogram(reference, bins = breakpoints)[0] / (len(reference) + eps) + eps
    cur_pct = np.histogram(current, bins = breakpoints)[0] / (len(current) + eps) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

# Kolmogorov - Smirnov (KS) test - https://medium.com/data-science/understanding-kolmogorov-smirnov-ks-tests-for-data-drift-on-profiled-data-5c8317796f78

def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """2 sample test, low p value (< 0.01) = distributions significantly different"""
    stat, p_value = stats.ks_2samp(reference, current)
    return float(stat), float(p_value)

def compute_score_distribution_psi(reference_scores: np.ndarray, current_scores: np.ndarray) -> float:
    """PSI on model output scores — proxy metric when ground truth labels are delayed."""
    return compute_psi(reference_scores, current_scores, n_bins=20)

# main detection

def detect_drift(window_id: int, current_df: pd.DataFrame, reference_df: pd.DataFrame) -> dict:
    config = load_config()
    thresholds = config["drift"]["thresholds"]
    results = {}
    any_triggered = False
 
    for feature in MONITOR_FEATURES:
        if feature not in current_df.columns or feature not in reference_df.columns:
            logger.warning(f"Feature {feature} missing in window {window_id} — schema drift!")
            results[feature] = {"psi": (999.0, thresholds["psi"], True), "ks_stat": (999.0, thresholds["ks_stat"], True)}
            any_triggered = True
            continue
 
        ref_vals = reference_df[feature].dropna().values
        cur_vals = current_df[feature].dropna().values
        psi_val = compute_psi(ref_vals, cur_vals)
        ks_stat, ks_pval = compute_ks(ref_vals, cur_vals)
        psi_triggered = psi_val > thresholds["psi"]
        ks_triggered = ks_pval < thresholds["ks_pvalue"]
 
        results[feature] = {
            "psi":      (psi_val,  thresholds["psi"],      psi_triggered),
            "ks_stat":  (ks_stat,  thresholds["ks_stat"],  ks_triggered),
            "ks_pvalue":(ks_pval,  thresholds["ks_pvalue"],ks_triggered),
        }
        if psi_triggered or ks_triggered:
            any_triggered = True
            logger.warning(f"Drift on {feature} window {window_id}: PSI={psi_val:.3f}, KS p={ks_pval:.4f}")
 
    results["_summary"] = {"drift_detected": any_triggered, "window_id": window_id}
    write_drift_metrics(window_id, {k: v for k, v in results.items() if not k.startswith("_")})
    return results
 
 
def check_performance_drift(window_id: int, y_true: np.ndarray, y_score: np.ndarray,
                             reference_scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr_arr, 0.05)
    metrics = {
        "auprc":              float(average_precision_score(y_true, y_score)),
        "auroc":              float(roc_auc_score(y_true, y_score)),
        "recall_at_5pct_fpr": float(tpr_arr[min(idx, len(tpr_arr) - 1)]),
        "score_dist_psi":     compute_score_distribution_psi(reference_scores, y_score),
    }
    write_performance_metrics(window_id, metrics)
    logger.info(f"Window {window_id} performance: {metrics}")
    return metrics
 
 
if __name__ == "__main__":
    init_db()
    reference = pd.read_parquet(WINDOWS_PATH / "reference.parquet")
    for i in range(1, 7):
        path = WINDOWS_PATH / f"window_{i}" / "data.parquet"
        if path.exists():
            current = pd.read_parquet(path)
            results = detect_drift(i, current, reference)
            print(f"\nWindow {i}: drift_detected={results['_summary']['drift_detected']}")
            for feat in MONITOR_FEATURES:
                if feat in results:
                    print(f"  {feat}: PSI={results[feat]['psi'][0]:.4f}")