import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from drift import compute_psi, compute_ks


class TestPSI:
    def test_identical_distributions_zero_psi(self):
        data = np.random.default_rng(42).normal(0, 1, 5000)
        assert compute_psi(data, data) < 0.01

    def test_heavily_shifted_distribution_high_psi(self):
        rng = np.random.default_rng(42)
        assert compute_psi(rng.normal(0,1,5000), rng.normal(3,1,5000)) > 0.20

    def test_moderate_shift_in_range(self):
        rng = np.random.default_rng(42)
        psi = compute_psi(rng.normal(0,1,5000), rng.normal(0.8,1.2,5000))
        assert 0.05 < psi < 0.60

    def test_psi_non_negative(self):
        rng = np.random.default_rng(42)
        assert compute_psi(rng.lognormal(0,1,3000), rng.lognormal(0.5,1.2,3000)) >= 0

    def test_amount_drift_from_simulate_exceeds_threshold(self):
        """Window 4 injects log-shift on Amount. PSI must exceed 0.20 retraining threshold."""
        rng = np.random.default_rng(0)
        ref = rng.lognormal(3.5, 2.0, 10000)
        log_ref = np.log1p(ref)
        cur = np.expm1(log_ref + log_ref.std())
        assert compute_psi(ref, cur) > 0.20


class TestKS:
    def test_same_distribution_high_pvalue(self):
        data = np.random.default_rng(42).normal(0, 1, 2000)
        _, p = compute_ks(data[:1000], data[1000:])
        assert p > 0.05

    def test_different_distribution_low_pvalue(self):
        rng = np.random.default_rng(42)
        _, p = compute_ks(rng.normal(0,1,2000), rng.normal(2,1,2000))
        assert p < 0.01

    def test_returns_valid_tuple(self):
        data = np.random.default_rng(42).normal(0, 1, 500)
        stat, p = compute_ks(data[:250], data[250:])
        assert 0 <= stat <= 1 and 0 <= p <= 1