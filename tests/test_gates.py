import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from gates import gate_auprc, gate_recall_at_fpr, gate_prediction_stability

CONFIG = {
    "gates": {
        "auprc":         {"min_improvement": 0.01, "min_auprc_floor": 0.75},
        "recall_at_fpr": {"target_fpr": 0.05, "max_regression": 0.02, "min_recall_floor": 0.70},
        "stability":     {"max_score_psi": 0.30},
        "latency":       {"max_latency_ms": 100},
    }
}


def make_scores(auprc_quality="good", seed=42):
    rng = np.random.default_rng(seed)
    n, fraud_rate = 2000, 0.02
    n_fraud = int(n * fraud_rate)
    y = np.array([1]*n_fraud + [0]*(n-n_fraud))
    if auprc_quality == "good":
        scores = np.concatenate([rng.beta(8,2,n_fraud), rng.beta(2,8,n-n_fraud)])
    else:
        scores = np.concatenate([rng.beta(3,3,n_fraud), rng.beta(3,3,n-n_fraud)])
    return y, scores


class TestAUPRCGate:
    def test_good_model_passes_floor(self):
        y, s = make_scores("good")
        assert gate_auprc(s, None, y, CONFIG).passed

    def test_poor_model_fails_floor(self):
        y, s = make_scores("poor")
        assert not gate_auprc(s, None, y, CONFIG).passed

    def test_worse_challenger_fails(self):
        y, s = make_scores("good")
        assert not gate_auprc(s - 0.05, s, y, CONFIG).passed

    def test_result_has_required_fields(self):
        y, s = make_scores()
        r = gate_auprc(s, None, y, CONFIG)
        assert r.gate_name and isinstance(r.passed, bool) and r.reason


class TestRecallAtFPRGate:
    def test_identical_model_no_regression(self):
        y, s = make_scores("good")
        assert gate_recall_at_fpr(s, s, y, CONFIG).passed

    def test_large_regression_fails(self):
        y, s = make_scores("good")
        assert not gate_recall_at_fpr(s - 0.3, s, y, CONFIG).passed

    def test_first_model_floor(self):
        y, s = make_scores("good")
        r = gate_recall_at_fpr(s, None, y, CONFIG)
        assert isinstance(r.passed, bool)


class TestPredictionStabilityGate:
    def test_no_champion_always_passes(self):
        assert gate_prediction_stability(None, None, np.array([]), CONFIG).passed

    def test_identical_model_passes(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (500, 5))
        y = (X[:,0] + rng.normal(0, 0.5, 500) > 0).astype(int)
        m = LogisticRegression(random_state=42).fit(X, y)
        assert gate_prediction_stability(m, m, X, CONFIG).passed