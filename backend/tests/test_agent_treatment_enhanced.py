"""Unit tests for enhanced TreatmentEffectsAgent methods - bootstrap CI and ensemble."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from app.agents import treatment
from app.agents.treatment import (
    _bootstrap_confidence_interval,
    _ensemble_estimate,
)


def _install_module(path: str, module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper to install mock modules."""
    parts = path.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, path, module)


class TestBootstrapConfidenceInterval:
    """Tests for _bootstrap_confidence_interval function."""

    def test_bootstrap_ci_basic(self, monkeypatch):
        """Test bootstrap CI returns valid bounds."""
        estimates = [0.18, 0.22, 0.19, 0.21, 0.20]
        estimate_idx = [0]

        class DummyEstimate:
            @property
            def value(self):
                idx = estimate_idx[0]
                estimate_idx[0] = (idx + 1) % len(estimates)
                return estimates[idx]

        class DummyModel:
            def __init__(self, data, treatment, outcome, common_causes=None, instruments=None):
                pass

            def identify_effect(self):
                return "estimand"

            def estimate_effect(self, estimand, method_name=None):
                return DummyEstimate()

        module = types.ModuleType("dowhy")
        module.CausalModel = DummyModel
        _install_module("dowhy", module, monkeypatch)

        df = pd.DataFrame({
            "treatment": [0, 1] * 100,
            "outcome": np.random.randn(200),
            "conf": np.random.randn(200),
        })

        ci_lower, ci_upper, samples = _bootstrap_confidence_interval(
            df, "treatment", "outcome", ["conf"],
            method_name="backdoor.propensity_score_matching",
            num_simulations=10,
            confidence_level=0.95,
        )

        assert ci_lower is not None
        assert ci_upper is not None
        assert samples is not None
        assert ci_lower < ci_upper
        assert len(samples) == 10

    def test_bootstrap_ci_iv_method(self, monkeypatch):
        """Test bootstrap CI works with IV method."""
        class DummyEstimate:
            value = 0.2

        class DummyModel:
            def __init__(self, data, treatment, outcome, common_causes=None, instruments=None):
                self.instruments = instruments

            def identify_effect(self):
                return "estimand"

            def estimate_effect(self, estimand, method_name=None):
                return DummyEstimate()

        module = types.ModuleType("dowhy")
        module.CausalModel = DummyModel
        _install_module("dowhy", module, monkeypatch)

        df = pd.DataFrame({
            "treatment": [0, 1] * 100,
            "outcome": np.random.randn(200),
            "instrument": np.random.randn(200),
        })

        ci_lower, ci_upper, samples = _bootstrap_confidence_interval(
            df, "treatment", "outcome", [],
            method_name="iv.instrumental_variable",
            num_simulations=10,
            confidence_level=0.95,
            instrument="instrument",
        )

        assert ci_lower is not None
        assert ci_upper is not None

    def test_bootstrap_ci_too_many_failures(self, monkeypatch):
        """Test bootstrap CI returns None when too many failures."""
        class DummyModel:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Always fails")

        module = types.ModuleType("dowhy")
        module.CausalModel = DummyModel
        _install_module("dowhy", module, monkeypatch)

        df = pd.DataFrame({
            "treatment": [0, 1] * 100,
            "outcome": np.random.randn(200),
        })

        ci_lower, ci_upper, samples = _bootstrap_confidence_interval(
            df, "treatment", "outcome", [],
            method_name="backdoor.propensity_score_matching",
            num_simulations=10,
        )

        assert ci_lower is None
        assert ci_upper is None
        assert samples is None

    def test_bootstrap_ci_no_dowhy(self, monkeypatch):
        """Test returns None when DoWhy not available."""
        monkeypatch.delitem(sys.modules, "dowhy", raising=False)

        df = pd.DataFrame({
            "treatment": [0, 1] * 50,
            "outcome": np.random.randn(100),
        })

        ci_lower, ci_upper, samples = _bootstrap_confidence_interval(
            df, "treatment", "outcome", [],
            method_name="backdoor.propensity_score_matching",
        )

        assert ci_lower is None
        assert ci_upper is None
        assert samples is None


class TestEnsembleEstimate:
    """Tests for _ensemble_estimate function."""

    def test_ensemble_two_methods(self):
        """Test ensemble with two methods."""
        method_results = [
            {
                "method": "propensity_matching",
                "ate": 0.20,
                "ate_ci_lower": 0.15,
                "ate_ci_upper": 0.25,
                "confidence_score": 0.8,
            },
            {
                "method": "doubly_robust",
                "ate": 0.22,
                "ate_ci_lower": 0.18,
                "ate_ci_upper": 0.26,
                "confidence_score": 0.85,
            },
        ]

        result = _ensemble_estimate(method_results)

        assert result is not None
        assert "ate" in result
        assert "ate_ci_lower" in result
        assert "ate_ci_upper" in result
        assert "weights" in result
        assert len(result["methods_used"]) == 2
        # Ensemble ATE should be between individual estimates
        assert 0.20 <= result["ate"] <= 0.22

    def test_ensemble_three_methods(self):
        """Test ensemble with three methods."""
        method_results = [
            {
                "method": "propensity_matching",
                "ate": 0.20,
                "ate_ci_lower": 0.15,
                "ate_ci_upper": 0.25,
                "confidence_score": 0.8,
            },
            {
                "method": "doubly_robust",
                "ate": 0.22,
                "ate_ci_lower": 0.18,
                "ate_ci_upper": 0.26,
                "confidence_score": 0.85,
            },
            {
                "method": "iv",
                "ate": 0.18,
                "ate_ci_lower": 0.10,
                "ate_ci_upper": 0.26,
                "confidence_score": 0.6,
            },
        ]

        result = _ensemble_estimate(method_results)

        assert result is not None
        assert len(result["methods_used"]) == 3
        # Doubly robust should have highest weight (highest confidence, narrow CI)
        assert result["weights"]["doubly_robust"] > result["weights"]["iv"]

    def test_ensemble_weighting_confidence(self):
        """Test that higher confidence gets higher weight."""
        method_results = [
            {
                "method": "low_conf",
                "ate": 0.20,
                "ate_ci_lower": 0.15,
                "ate_ci_upper": 0.25,
                "confidence_score": 0.5,  # Low confidence
            },
            {
                "method": "high_conf",
                "ate": 0.22,
                "ate_ci_lower": 0.17,
                "ate_ci_upper": 0.27,
                "confidence_score": 0.9,  # High confidence
            },
        ]

        result = _ensemble_estimate(method_results)

        assert result is not None
        assert result["weights"]["high_conf"] > result["weights"]["low_conf"]

    def test_ensemble_weighting_ci_width(self):
        """Test that narrower CI gets higher weight."""
        method_results = [
            {
                "method": "wide_ci",
                "ate": 0.20,
                "ate_ci_lower": 0.05,
                "ate_ci_upper": 0.35,  # Wide CI (0.30)
                "confidence_score": 0.8,
            },
            {
                "method": "narrow_ci",
                "ate": 0.22,
                "ate_ci_lower": 0.19,
                "ate_ci_upper": 0.25,  # Narrow CI (0.06)
                "confidence_score": 0.8,
            },
        ]

        result = _ensemble_estimate(method_results)

        assert result is not None
        assert result["weights"]["narrow_ci"] > result["weights"]["wide_ci"]

    def test_ensemble_insufficient_methods(self):
        """Test ensemble returns None with less than 2 methods."""
        method_results = [
            {
                "method": "only_one",
                "ate": 0.20,
                "ate_ci_lower": 0.15,
                "ate_ci_upper": 0.25,
                "confidence_score": 0.8,
            },
        ]

        result = _ensemble_estimate(method_results)

        assert result is None

    def test_ensemble_missing_ci(self):
        """Test ensemble handles missing CI gracefully."""
        method_results = [
            {
                "method": "with_ci",
                "ate": 0.20,
                "ate_ci_lower": 0.15,
                "ate_ci_upper": 0.25,
                "confidence_score": 0.8,
            },
            {
                "method": "without_ci",
                "ate": 0.22,
                "ate_ci_lower": None,
                "ate_ci_upper": None,
                "confidence_score": 0.7,
            },
        ]

        result = _ensemble_estimate(method_results)

        assert result is not None
        # Method with CI should have higher weight
        assert result["weights"]["with_ci"] > result["weights"]["without_ci"]

    def test_ensemble_confidence_score(self):
        """Test ensemble confidence score is weighted average."""
        method_results = [
            {
                "method": "method1",
                "ate": 0.20,
                "ate_ci_lower": 0.15,
                "ate_ci_upper": 0.25,
                "confidence_score": 0.6,
            },
            {
                "method": "method2",
                "ate": 0.20,
                "ate_ci_lower": 0.15,
                "ate_ci_upper": 0.25,
                "confidence_score": 0.8,
            },
        ]

        result = _ensemble_estimate(method_results)

        assert result is not None
        # Ensemble confidence should be between individual scores
        assert 0.6 <= result["confidence_score"] <= 0.8

    def test_ensemble_assumptions_checked(self):
        """Test ensemble includes method info in assumptions_checked."""
        method_results = [
            {
                "method": "propensity_matching",
                "ate": 0.20,
                "ate_ci_lower": 0.15,
                "ate_ci_upper": 0.25,
                "confidence_score": 0.8,
            },
            {
                "method": "doubly_robust",
                "ate": 0.22,
                "ate_ci_lower": 0.18,
                "ate_ci_upper": 0.26,
                "confidence_score": 0.85,
            },
        ]

        result = _ensemble_estimate(method_results)

        assert result is not None
        assert result["assumptions_checked"]["ensemble"] is True
        assert "propensity_matching" in result["assumptions_checked"]["methods"]
        assert "doubly_robust" in result["assumptions_checked"]["methods"]


class TestUpdatedPSMEstimation:
    """Tests for updated PSM estimation with bootstrap CI."""

    def test_psm_returns_bootstrap_ci(self, monkeypatch):
        """Test that PSM returns bootstrap CI information."""
        from app.agents.treatment import _estimate_psm

        class DummyEstimate:
            value = 0.2

            def get_confidence_intervals(self):
                return [(0.15, 0.25)]

        class DummyModel:
            def __init__(self, data, treatment, outcome, common_causes=None):
                pass

            def identify_effect(self):
                return "estimand"

            def estimate_effect(self, estimand, method_name=None):
                return DummyEstimate()

        module = types.ModuleType("dowhy")
        module.CausalModel = DummyModel
        _install_module("dowhy", module, monkeypatch)

        df = pd.DataFrame({
            "treatment": [0, 1] * 100,
            "outcome": np.random.randn(200),
            "conf": np.random.randn(200),
        })

        result = _estimate_psm(df, "treatment", "outcome", ["conf"])

        assert result is not None
        assert "ate" in result
        assert "confidence_interval" in result
        assert result["confidence_interval"]["method"] == "bootstrap"
        assert "confidence_score" in result


class TestUpdatedDoublyRobustEstimation:
    """Tests for updated doubly robust estimation with bootstrap CI."""

    def test_doubly_robust_returns_bootstrap_ci(self, monkeypatch):
        """Test that doubly robust returns bootstrap CI information."""
        from app.agents.treatment import _estimate_doubly_robust

        class DummyDML:
            def __init__(self, model_y=None, model_t=None):
                pass

            def fit(self, y, t, X=None):
                return self

            def ate(self, X=None):
                return 0.2 + np.random.randn() * 0.02

        econml_module = types.ModuleType("econml")
        dml_module = types.ModuleType("econml.dml")
        dml_module.LinearDML = DummyDML
        econml_module.dml = dml_module

        sklearn_module = types.ModuleType("sklearn")
        linear_model_module = types.ModuleType("sklearn.linear_model")
        linear_model_module.LinearRegression = lambda: None
        linear_model_module.LogisticRegression = lambda max_iter=1000: None
        sklearn_module.linear_model = linear_model_module

        _install_module("econml", econml_module, monkeypatch)
        _install_module("econml.dml", dml_module, monkeypatch)
        _install_module("sklearn", sklearn_module, monkeypatch)
        _install_module("sklearn.linear_model", linear_model_module, monkeypatch)

        df = pd.DataFrame({
            "treatment": [0, 1] * 100,
            "outcome": np.random.randn(200),
            "conf": np.random.randn(200),
        })

        result = _estimate_doubly_robust(df, "treatment", "outcome", ["conf"])

        assert result is not None
        assert "ate" in result
        assert "confidence_interval" in result
        assert result["confidence_interval"]["method"] == "bootstrap"
        assert "confidence_score" in result


class TestUpdatedIVEstimation:
    """Tests for updated IV estimation with first-stage F-statistic."""

    def test_iv_returns_first_stage_f(self, monkeypatch):
        """Test that IV returns first-stage F-statistic."""
        from app.agents.treatment import _estimate_iv

        class DummyEstimate:
            value = 0.2

        class DummyModel:
            def __init__(self, data, treatment, outcome, common_causes=None, instruments=None):
                pass

            def identify_effect(self):
                return "estimand"

            def estimate_effect(self, estimand, method_name=None):
                return DummyEstimate()

        module = types.ModuleType("dowhy")
        module.CausalModel = DummyModel
        _install_module("dowhy", module, monkeypatch)

        # Create data with a correlated instrument
        np.random.seed(42)
        instrument = np.random.randn(200)
        treatment = (instrument > 0).astype(int)
        outcome = treatment * 0.5 + np.random.randn(200) * 0.1

        df = pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
            "instrument": instrument,
        })

        # Mock _select_instrument to return our instrument
        monkeypatch.setattr(treatment, "_select_instrument", lambda *args: "instrument")

        result = _estimate_iv(df, "treatment", "outcome", [])

        assert result is not None
        assert "ate" in result
        assert "first_stage_f_statistic" in result["assumptions_checked"]
        assert result["assumptions_checked"]["instrument"] == "instrument"
