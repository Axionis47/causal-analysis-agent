"""Unit tests for enhanced ValidationAgent methods - sensitivity analysis and assumption checks."""

from __future__ import annotations

import sys
import types
import uuid

import numpy as np
import pandas as pd
import pytest

from app.agents import validation
from app.agents.validation import (
    _run_sensitivity_linear,
    _run_sensitivity_evalue,
    _check_positivity,
    _check_unconfoundedness,
    _check_sutva,
)


def _install_module(path: str, module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper to install mock modules."""
    parts = path.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, path, module)


class TestSensitivityLinear:
    """Tests for _run_sensitivity_linear function."""

    def test_sensitivity_linear_passed(self, monkeypatch):
        """Test sensitivity analysis passes with robust effect."""
        class DummyRefute:
            new_effect = 0.18  # Only small reduction from original

        class DummyEstimate:
            value = 0.2

            def get_confidence_intervals(self):
                return [(0.1, 0.3)]

        class DummyModel:
            def __init__(self, data, treatment, outcome, common_causes=None):
                pass

            def identify_effect(self):
                return "estimand"

            def estimate_effect(self, estimand, method_name=None):
                return DummyEstimate()

            def refute_estimate(self, estimand, estimate, method_name=None, **kwargs):
                return DummyRefute()

        module = types.ModuleType("dowhy")
        module.CausalModel = DummyModel
        _install_module("dowhy", module, monkeypatch)

        df = pd.DataFrame({
            "treatment": [0, 1, 0, 1] * 25,
            "outcome": [1, 2, 1.5, 2.5] * 25,
            "conf": [0.1, 0.2, 0.15, 0.25] * 25,
        })

        result = _run_sensitivity_linear(df, "treatment", "outcome", ["conf"])

        assert result is not None
        assert result["method"] == "sensitivity_linear"
        assert "robustness_value" in result["details"]

    def test_sensitivity_linear_no_dowhy(self, monkeypatch):
        """Test returns None when DoWhy not available."""
        monkeypatch.delitem(sys.modules, "dowhy", raising=False)

        df = pd.DataFrame({"treatment": [0, 1], "outcome": [1, 2]})
        result = _run_sensitivity_linear(df, "treatment", "outcome", [])

        assert result is None


class TestSensitivityEvalue:
    """Tests for _run_sensitivity_evalue function."""

    def test_evalue_binary_outcome(self, monkeypatch):
        """Test E-value computation for binary outcome."""
        class DummyEstimate:
            value = 0.5

            def get_confidence_intervals(self):
                return [(0.3, 0.7)]

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

        # Binary outcome
        df = pd.DataFrame({
            "treatment": [0, 1, 0, 1] * 25,
            "outcome": [0, 1, 0, 1] * 25,
        })

        result = _run_sensitivity_evalue(df, "treatment", "outcome", [])

        assert result is not None
        assert result["method"] == "sensitivity_evalue"
        assert result["details"]["is_binary_outcome"] is True
        assert result["details"]["evalue"] is not None

    def test_evalue_continuous_outcome(self, monkeypatch):
        """Test E-value computation for continuous outcome."""
        class DummyEstimate:
            value = 0.5

            def get_confidence_intervals(self):
                return [(0.3, 0.7)]

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

        # Continuous outcome
        df = pd.DataFrame({
            "treatment": [0, 1, 0, 1] * 25,
            "outcome": np.random.randn(100),
        })

        result = _run_sensitivity_evalue(df, "treatment", "outcome", [])

        assert result is not None
        assert result["details"]["is_binary_outcome"] is False


class TestPositivityCheck:
    """Tests for _check_positivity function."""

    def test_positivity_passed(self):
        """Test positivity check passes with good overlap."""
        np.random.seed(42)
        n = 200
        conf = np.random.randn(n)
        # Treatment probability varies smoothly with confounder
        prob = 1 / (1 + np.exp(-conf))
        treatment = (np.random.rand(n) < prob).astype(int)
        outcome = treatment * 0.5 + conf * 0.3 + np.random.randn(n) * 0.1

        df = pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
            "conf": conf,
        })

        result = _check_positivity(df, "treatment", "outcome", ["conf"])

        assert result is not None
        assert result["method"] == "positivity"
        assert "violation_rate" in result["details"]
        assert "overlap_region" in result["details"]

    def test_positivity_violation(self):
        """Test positivity check detects violation with extreme propensities."""
        # Create data where treatment is deterministic based on confounder
        conf = np.array([0] * 50 + [1] * 50)
        treatment = conf.copy()  # Treatment = conf exactly
        outcome = treatment * 0.5 + np.random.randn(100) * 0.1

        df = pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
            "conf": conf.astype(float),
        })

        result = _check_positivity(df, "treatment", "outcome", ["conf"])

        assert result is not None
        # Should detect violation due to perfect separation
        assert result["details"]["violation_rate"] > 0 or result["details"]["overlap_region"] < 1.0

    def test_positivity_non_binary_treatment(self):
        """Test positivity check handles non-binary treatment."""
        df = pd.DataFrame({
            "treatment": [0, 1, 2, 3] * 25,
            "outcome": np.random.randn(100),
            "conf": np.random.randn(100),
        })

        result = _check_positivity(df, "treatment", "outcome", ["conf"])

        assert result is not None
        assert result["passed"] is True
        assert "Non-binary" in result["details"].get("note", "")


class TestUnconfoundednessCheck:
    """Tests for _check_unconfoundedness function."""

    def test_unconfoundedness_balanced(self):
        """Test unconfoundedness check with balanced confounders."""
        np.random.seed(42)
        n = 200
        treatment = np.random.randint(0, 2, n)
        # Confounder is independent of treatment
        conf = np.random.randn(n)
        outcome = treatment * 0.5 + conf * 0.3 + np.random.randn(n) * 0.1

        df = pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
            "conf": conf,
        })

        result = _check_unconfoundedness(df, "treatment", "outcome", ["conf"])

        assert result is not None
        assert result["method"] == "unconfoundedness"
        assert result["details"]["imbalance_rate"] < 0.5

    def test_unconfoundedness_imbalanced(self):
        """Test unconfoundedness check detects imbalance."""
        n = 200
        # Create data where confounder differs by treatment
        treatment = np.array([0] * 100 + [1] * 100)
        conf = np.array([0.0] * 100 + [2.0] * 100)  # Very different means
        outcome = treatment * 0.5 + conf * 0.3 + np.random.randn(n) * 0.1

        df = pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
            "conf": conf,
        })

        result = _check_unconfoundedness(df, "treatment", "outcome", ["conf"])

        assert result is not None
        assert len(result["details"]["imbalanced_confounders"]) > 0
        assert "conf" in result["details"]["imbalanced_confounders"]

    def test_unconfoundedness_no_confounders(self):
        """Test unconfoundedness check with no confounders."""
        df = pd.DataFrame({
            "treatment": [0, 1] * 50,
            "outcome": np.random.randn(100),
        })

        result = _check_unconfoundedness(df, "treatment", "outcome", [])

        assert result is not None
        assert result["passed"] is True
        assert "No confounders" in result["details"].get("note", "")


class TestSUTVACheck:
    """Tests for _check_sutva function."""

    def test_sutva_no_clustering(self):
        """Test SUTVA check passes with no clustering."""
        np.random.seed(42)
        df = pd.DataFrame({
            "treatment": np.random.randint(0, 2, 100),
            "outcome": np.random.randn(100),
            "x1": np.random.randn(100),
        })

        result = _check_sutva(df, "treatment", "outcome", ["x1"])

        assert result is not None
        assert result["method"] == "sutva"
        assert result["details"]["clustering_detected"] is False
        assert len(result["details"]["spillover_indicators"]) == 0

    def test_sutva_temporal_clustering(self):
        """Test SUTVA check detects temporal clustering."""
        np.random.seed(42)
        # Create time-correlated treatment
        time = np.arange(100)
        treatment = (np.sin(time / 10) > 0).astype(int)  # Periodic treatment
        outcome = treatment * 0.5 + np.random.randn(100) * 0.1

        df = pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
            "timestamp": time,
        })

        result = _check_sutva(df, "treatment", "outcome", [])

        assert result is not None
        assert result["details"]["clustering_detected"] is True
        assert len(result["details"]["time_columns"]) > 0

    def test_sutva_location_clustering(self):
        """Test SUTVA check detects location clustering."""
        # Create location-clustered treatment
        location = np.array(["A"] * 50 + ["B"] * 50)
        treatment = np.array([1] * 50 + [0] * 50)  # All treated in A, none in B
        outcome = treatment * 0.5 + np.random.randn(100) * 0.1

        df = pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
            "location": location,
        })

        result = _check_sutva(df, "treatment", "outcome", [])

        assert result is not None
        assert result["details"]["clustering_detected"] is True
        assert len(result["details"]["location_columns"]) > 0
        # Should detect high variance in treatment rates
        if result["details"]["location_clustering"]:
            assert result["details"]["location_clustering"]["treatment_rate_variance"] > 0


class TestValidationAgentIntegration:
    """Integration tests for ValidationAgent with new methods."""

    @pytest.mark.asyncio
    async def test_validation_runs_all_checks(self, db_session, monkeypatch):
        """Test that ValidationAgent runs all new check methods."""
        from tests.fixtures.agent_fixtures import create_analysis, create_dataset, create_treatment_effect
        from tests.fixtures.mock_helpers import make_session_context

        analysis = await create_analysis(db_session)
        dataset = await create_dataset(db_session, analysis_id=analysis.id)
        await create_treatment_effect(db_session, analysis_id=analysis.id)

        sensitivity_called = {"linear": False, "evalue": False}
        assumption_called = {"positivity": False, "unconfoundedness": False, "sutva": False}

        def fake_sensitivity_linear(*args, **kwargs):
            sensitivity_called["linear"] = True
            return {
                "method": "sensitivity_linear",
                "passed": True,
                "confidence_score": 0.7,
                "details": {"robustness_value": 0.15},
                "recommendations": [],
            }

        def fake_sensitivity_evalue(*args, **kwargs):
            sensitivity_called["evalue"] = True
            return {
                "method": "sensitivity_evalue",
                "passed": True,
                "confidence_score": 0.8,
                "details": {"evalue": 2.0},
                "recommendations": [],
            }

        def fake_positivity(*args, **kwargs):
            assumption_called["positivity"] = True
            return {
                "method": "positivity",
                "passed": True,
                "confidence_score": 0.9,
                "details": {"violation_rate": 0.02},
                "recommendations": [],
            }

        def fake_unconfoundedness(*args, **kwargs):
            assumption_called["unconfoundedness"] = True
            return {
                "method": "unconfoundedness",
                "passed": True,
                "confidence_score": 0.85,
                "details": {"imbalance_rate": 0.1},
                "recommendations": [],
            }

        def fake_sutva(*args, **kwargs):
            assumption_called["sutva"] = True
            return {
                "method": "sutva",
                "passed": True,
                "confidence_score": 0.9,
                "details": {"clustering_detected": False},
                "recommendations": [],
            }

        def fake_refutation(*args, **kwargs):
            return {
                "method": "refutation",
                "passed": True,
                "confidence_score": 0.7,
                "details": {"p_value": 0.8},
                "recommendations": [],
            }

        monkeypatch.setattr(validation, "get_session_context", make_session_context(db_session))
        monkeypatch.setattr(validation, "_load_dataframe", lambda *_: pd.DataFrame())
        monkeypatch.setattr(validation, "_run_refutation", fake_refutation)
        monkeypatch.setattr(validation, "_run_sensitivity_linear", fake_sensitivity_linear)
        monkeypatch.setattr(validation, "_run_sensitivity_evalue", fake_sensitivity_evalue)
        monkeypatch.setattr(validation, "_check_positivity", fake_positivity)
        monkeypatch.setattr(validation, "_check_unconfoundedness", fake_unconfoundedness)
        monkeypatch.setattr(validation, "_check_sutva", fake_sutva)

        from app.agents.validation import ValidationAgent
        state = {"analysis_id": str(analysis.id), "dataset_id": str(dataset.id)}

        result = await ValidationAgent()._run(state)

        assert result.message == "Validation complete"
        assert sensitivity_called["linear"] is True
        assert sensitivity_called["evalue"] is True
        assert assumption_called["positivity"] is True
        assert assumption_called["unconfoundedness"] is True
        assert assumption_called["sutva"] is True
        assert result.outputs["sensitivity_count"] >= 1
        assert result.outputs["assumption_count"] >= 1
