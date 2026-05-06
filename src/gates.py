import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import logging
import yaml
from dataclasses import dataclass, field

import features as feat_module

logger = logging.getLogger(__name__)
CONFIG_PATH = Path("configs/config.yaml")
WINDOWS_PATH = Path("data/windows")


def load_config() -> dict:
    with open(CONFIG_PATH) as f: return yaml.safe_load(f)


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    reason: str
    challenger_value: float = 0.0
    champion_value: float = 0.0
    threshold: float = 0.0


@dataclass
class ValidationReport:
    challenger_run_id: str
    champion_run_id: str | None
    gate_results: list[GateResult] = field(default_factory=list)
    overall_passed: bool = False
    promotion_decision: str = "rejected"

    def summary(self) -> str:
        lines = ["=== Validation Report ===",
                 f"Challenger: {self.challenger_run_id}",
                 f"Champion:   {self.champion_run_id}",
                 f"Decision:   {self.promotion_decision.upper()}", ""]
        for g in self.gate_results:
            lines.append(f"  [{'PASS' if g.passed else 'FAIL'}] {g.gate_name}: {g.reason}")
        return "\n".join(lines)


def get_champion_run_id() -> str | None:
    client = mlflow.MlflowClient()
    try:
        versions = client.get_latest_versions("fraud-detector", stages=["Production"])
        if versions: return versions[0].run_id
    except Exception as e:
        logger.warning(f"Could not fetch champion: {e}")
    return None


def load_holdout(window_id: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Fixed holdout: last 20% of window 3. Stable baseline, never sees injected drift."""
    df = pd.read_parquet(WINDOWS_PATH / f"window_{window_id}" / "data.parquet")
    return feat_module.transform(df.iloc[int(len(df) * 0.8):])


def gate_auprc(challenger_score, champion_score, y_true, config) -> GateResult:
    """
    Primary gate: AUPRC beats champion by min_improvement margin.
    AUPRC is the correct metric under 0.17% class imbalance —
    accuracy and ROC-AUC are both misleading in this regime.
    """
    from sklearn.metrics import average_precision_score
    gc = config["gates"]["auprc"]
    c_auprc = average_precision_score(y_true, challenger_score)
    if champion_score is None:
        passed = c_auprc >= gc["min_auprc_floor"]
        return GateResult("AUPRC (vs floor)", passed,
                          f"Challenger={c_auprc:.4f}, floor={gc['min_auprc_floor']}",
                          challenger_value=c_auprc, threshold=gc["min_auprc_floor"])
    ch_auprc = average_precision_score(y_true, champion_score)
    imp = c_auprc - ch_auprc
    return GateResult("AUPRC (vs champion)", imp >= gc["min_improvement"],
                      f"Challenger={c_auprc:.4f}, Champion={ch_auprc:.4f}, improvement={imp:.4f} (required={gc['min_improvement']})",
                      challenger_value=c_auprc, champion_value=ch_auprc, threshold=gc["min_improvement"])


def gate_recall_at_fpr(challenger_score, champion_score, y_true, config) -> GateResult:
    """Business metric gate: recall at fixed 5% FPR must not regress vs champion."""
    from sklearn.metrics import roc_curve
    gc = config["gates"]["recall_at_fpr"]
    def _recall(scores):
        fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
        return float(tpr_arr[min(np.searchsorted(fpr_arr, gc["target_fpr"]), len(tpr_arr)-1)])
    c_recall = _recall(challenger_score)
    if champion_score is None:
        return GateResult(f"Recall@{int(gc['target_fpr']*100)}%FPR (vs floor)",
                          c_recall >= gc["min_recall_floor"],
                          f"Challenger={c_recall:.4f}, floor={gc['min_recall_floor']}",
                          challenger_value=c_recall, threshold=gc["min_recall_floor"])
    ch_recall = _recall(champion_score)
    reg = ch_recall - c_recall
    return GateResult(f"Recall@{int(gc['target_fpr']*100)}%FPR (no regression)",
                      reg <= gc["max_regression"],
                      f"Challenger={c_recall:.4f}, Champion={ch_recall:.4f}, regression={reg:.4f} (max={gc['max_regression']})",
                      challenger_value=c_recall, champion_value=ch_recall, threshold=gc["max_regression"])


def gate_prediction_stability(challenger_model, champion_model, X_ref, config) -> GateResult:
    """PSI on output score distributions. High PSI = training ran differently, not just improved."""
    if champion_model is None:
        return GateResult("Prediction Stability", True, "No champion — skipping.")
    from drift import compute_psi
    gc = config["gates"]["stability"]
    psi = compute_psi(champion_model.predict_proba(X_ref)[:,1],
                      challenger_model.predict_proba(X_ref)[:,1], n_bins=20)
    return GateResult("Prediction Stability (PSI)", psi <= gc["max_score_psi"],
                      f"Score PSI={psi:.4f} (max={gc['max_score_psi']})",
                      challenger_value=psi, threshold=gc["max_score_psi"])


def gate_latency(challenger_model, X_sample, config) -> GateResult:
    """Median single-prediction latency must stay under threshold."""
    import time
    gc = config["gates"]["latency"]
    for _ in range(5): challenger_model.predict_proba(X_sample[:10])
    times = []
    for _ in range(50):
        t = time.perf_counter()
        challenger_model.predict_proba(X_sample[:1])
        times.append(time.perf_counter() - t)
    ms = np.median(times) * 1000
    return GateResult("Inference Latency", ms <= gc["max_latency_ms"],
                      f"Median={ms:.2f}ms (max={gc['max_latency_ms']}ms)",
                      challenger_value=ms, threshold=gc["max_latency_ms"])


def validate_and_promote(challenger_run_id: str) -> ValidationReport:
    config = load_config()
    client = mlflow.MlflowClient()
    champion_run_id = get_champion_run_id()
    report = ValidationReport(challenger_run_id=challenger_run_id, champion_run_id=champion_run_id)

    challenger_model = mlflow.sklearn.load_model(f"runs:/{challenger_run_id}/model")
    champion_model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model") if champion_run_id else None
    X_hold, y_hold = load_holdout()
    c_scores = challenger_model.predict_proba(X_hold)[:,1]
    ch_scores = champion_model.predict_proba(X_hold)[:,1] if champion_model else None

    report.gate_results = [
        gate_auprc(c_scores, ch_scores, y_hold, config),
        gate_recall_at_fpr(c_scores, ch_scores, y_hold, config),
        gate_prediction_stability(challenger_model, champion_model, X_hold, config),
        gate_latency(challenger_model, X_hold, config),
    ]
    report.overall_passed = all(g.passed for g in report.gate_results)

    if report.overall_passed:
        for v in client.get_latest_versions("fraud-detector", stages=["None", "Staging"]):
            if v.run_id == challenger_run_id:
                client.transition_model_version_stage("fraud-detector", v.version, "Production")
                if champion_run_id:
                    for cv in client.get_latest_versions("fraud-detector", stages=["Production"]):
                        if cv.run_id == champion_run_id:
                            client.transition_model_version_stage("fraud-detector", cv.version, "Archived")
                report.promotion_decision = "promoted"
                break
    else:
        report.promotion_decision = "rejected"

    print(report.summary())
    return report