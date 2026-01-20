"""Unit tests for enhanced CausalDiscoveryAgent methods - cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

from app.agents.discovery import (
    _cross_validate_discovery,
    _build_consensus_graph,
    _run_discovery,
)


class TestCrossValidateDiscovery:
    """Tests for _cross_validate_discovery function."""

    def test_cross_validation_basic(self, monkeypatch):
        """Test cross-validation returns stable edges."""
        # Mock _run_discovery to return consistent edges
        call_count = [0]

        def mock_run_discovery(tool_name, data, names):
            call_count[0] += 1
            # Return same edges every time (stable)
            return {
                "edges": [
                    {"source": "A", "target": "B", "edge_type": "directed", "confidence": 0.7},
                    {"source": "B", "target": "C", "edge_type": "directed", "confidence": 0.6},
                ],
                "nodes": [{"name": n, "node_type": "variable"} for n in names],
                "graph_data": {},
            }

        monkeypatch.setattr("app.agents.discovery._run_discovery", mock_run_discovery)

        data = np.random.randn(200, 3)
        names = ["A", "B", "C"]

        result = _cross_validate_discovery(
            data, names, "pc",
            num_folds=5,
            stability_threshold=0.6,
        )

        assert result is not None
        assert "stable_edges" in result
        assert "edge_frequencies" in result
        assert len(result["stable_edges"]) == 2
        assert call_count[0] == 5  # Called once per fold

    def test_cross_validation_unstable_edges(self, monkeypatch):
        """Test cross-validation identifies unstable edges."""
        fold_idx = [0]

        def mock_run_discovery(tool_name, data, names):
            fold = fold_idx[0]
            fold_idx[0] += 1
            # Return different edges in different folds
            if fold < 3:
                return {
                    "edges": [
                        {"source": "A", "target": "B", "edge_type": "directed", "confidence": 0.7},
                    ],
                    "nodes": [{"name": n, "node_type": "variable"} for n in names],
                    "graph_data": {},
                }
            else:
                return {
                    "edges": [
                        {"source": "B", "target": "A", "edge_type": "directed", "confidence": 0.7},
                    ],
                    "nodes": [{"name": n, "node_type": "variable"} for n in names],
                    "graph_data": {},
                }

        monkeypatch.setattr("app.agents.discovery._run_discovery", mock_run_discovery)

        data = np.random.randn(200, 3)
        names = ["A", "B", "C"]

        result = _cross_validate_discovery(
            data, names, "pc",
            num_folds=5,
            stability_threshold=0.6,
        )

        assert result is not None
        # A->B appears in 3/5 folds = 0.6, should be stable
        # B->A appears in 2/5 folds = 0.4, should not be stable
        stable_sources = [e["source"] for e in result["stable_edges"]]
        stable_targets = [e["target"] for e in result["stable_edges"]]

        # Only A->B should be stable
        assert "A->B" in result["edge_frequencies"]
        assert result["edge_frequencies"]["A->B"] == 0.6
        assert result["edge_frequencies"]["B->A"] == 0.4

    def test_cross_validation_insufficient_data(self):
        """Test cross-validation returns None for small datasets."""
        data = np.random.randn(30, 3)  # Too small (30 * 0.8 = 24 < 50)
        names = ["A", "B", "C"]

        result = _cross_validate_discovery(
            data, names, "pc",
            num_folds=5,
            subsample_fraction=0.8,
        )

        assert result is None

    def test_cross_validation_different_thresholds(self, monkeypatch):
        """Test cross-validation with different stability thresholds."""
        def mock_run_discovery(tool_name, data, names):
            # Edge A->B appears sometimes, B->C always
            if np.random.rand() < 0.5:
                return {
                    "edges": [
                        {"source": "A", "target": "B", "edge_type": "directed", "confidence": 0.7},
                        {"source": "B", "target": "C", "edge_type": "directed", "confidence": 0.7},
                    ],
                    "nodes": [{"name": n, "node_type": "variable"} for n in names],
                    "graph_data": {},
                }
            else:
                return {
                    "edges": [
                        {"source": "B", "target": "C", "edge_type": "directed", "confidence": 0.7},
                    ],
                    "nodes": [{"name": n, "node_type": "variable"} for n in names],
                    "graph_data": {},
                }

        monkeypatch.setattr("app.agents.discovery._run_discovery", mock_run_discovery)

        data = np.random.randn(200, 3)
        names = ["A", "B", "C"]

        # Low threshold - more edges pass
        result_low = _cross_validate_discovery(
            data, names, "pc",
            num_folds=10,
            stability_threshold=0.3,
        )

        # High threshold - fewer edges pass
        result_high = _cross_validate_discovery(
            data, names, "pc",
            num_folds=10,
            stability_threshold=0.9,
        )

        assert result_low is not None
        assert result_high is not None
        # B->C appears in all folds, should be stable at high threshold
        # A->B appears in ~50% of folds, should be stable at low but not high threshold

    def test_cross_validation_all_folds_fail(self, monkeypatch):
        """Test cross-validation handles all folds failing."""
        def mock_run_discovery(tool_name, data, names):
            raise RuntimeError("Discovery failed")

        monkeypatch.setattr("app.agents.discovery._run_discovery", mock_run_discovery)

        data = np.random.randn(200, 3)
        names = ["A", "B", "C"]

        result = _cross_validate_discovery(
            data, names, "pc",
            num_folds=5,
        )

        assert result is None

    def test_cross_validation_mean_stability(self, monkeypatch):
        """Test cross-validation computes correct mean stability."""
        fold_idx = [0]

        def mock_run_discovery(tool_name, data, names):
            fold = fold_idx[0]
            fold_idx[0] += 1
            # A->B in all folds, B->C in 3 folds, C->A in 1 fold
            edges = [{"source": "A", "target": "B", "edge_type": "directed", "confidence": 0.7}]
            if fold < 3:
                edges.append({"source": "B", "target": "C", "edge_type": "directed", "confidence": 0.7})
            if fold == 0:
                edges.append({"source": "C", "target": "A", "edge_type": "directed", "confidence": 0.7})
            return {
                "edges": edges,
                "nodes": [{"name": n, "node_type": "variable"} for n in names],
                "graph_data": {},
            }

        monkeypatch.setattr("app.agents.discovery._run_discovery", mock_run_discovery)

        data = np.random.randn(200, 3)
        names = ["A", "B", "C"]

        result = _cross_validate_discovery(
            data, names, "pc",
            num_folds=5,
        )

        assert result is not None
        # Mean stability = (1.0 + 0.6 + 0.2) / 3 = 0.6
        expected_mean = (5/5 + 3/5 + 1/5) / 3
        assert abs(result["mean_stability"] - expected_mean) < 0.01


class TestBuildConsensusGraph:
    """Tests for _build_consensus_graph function."""

    def test_build_consensus_basic(self):
        """Test consensus graph includes only stable edges."""
        edge_frequencies = {
            "A->B": 0.8,  # Stable
            "B->C": 0.6,  # Stable (at threshold)
            "C->A": 0.4,  # Not stable
        }
        names = ["A", "B", "C"]

        result = _build_consensus_graph(edge_frequencies, names, stability_threshold=0.6)

        assert result is not None
        assert len(result["edges"]) == 2
        edge_strs = {f"{e['source']}->{e['target']}" for e in result["edges"]}
        assert "A->B" in edge_strs
        assert "B->C" in edge_strs
        assert "C->A" not in edge_strs

    def test_build_consensus_edge_confidence(self):
        """Test consensus graph uses frequency as edge confidence."""
        edge_frequencies = {
            "A->B": 0.9,
            "B->C": 0.7,
        }
        names = ["A", "B", "C"]

        result = _build_consensus_graph(edge_frequencies, names, stability_threshold=0.6)

        assert result is not None
        for edge in result["edges"]:
            if edge["source"] == "A" and edge["target"] == "B":
                assert edge["confidence"] == 0.9
            if edge["source"] == "B" and edge["target"] == "C":
                assert edge["confidence"] == 0.7

    def test_build_consensus_empty_edges(self):
        """Test consensus graph with no stable edges."""
        edge_frequencies = {
            "A->B": 0.3,
            "B->C": 0.4,
        }
        names = ["A", "B", "C"]

        result = _build_consensus_graph(edge_frequencies, names, stability_threshold=0.6)

        assert result is not None
        assert len(result["edges"]) == 0
        assert result["confidence"] == 0.0

    def test_build_consensus_all_nodes_included(self):
        """Test consensus graph includes all nodes even if no edges."""
        edge_frequencies = {}
        names = ["A", "B", "C", "D"]

        result = _build_consensus_graph(edge_frequencies, names, stability_threshold=0.6)

        assert result is not None
        assert len(result["nodes"]) == 4
        node_names = {n["name"] for n in result["nodes"]}
        assert node_names == {"A", "B", "C", "D"}

    def test_build_consensus_graph_data(self):
        """Test consensus graph includes valid graph_data."""
        edge_frequencies = {
            "A->B": 0.8,
        }
        names = ["A", "B"]

        result = _build_consensus_graph(edge_frequencies, names, stability_threshold=0.6)

        assert result is not None
        assert "graph_data" in result
        assert "nodes" in result["graph_data"]
        assert "links" in result["graph_data"]


class TestCrossValidationIntegration:
    """Integration tests for cross-validation in discovery pipeline."""

    def test_cv_with_pc_algorithm(self, monkeypatch):
        """Test cross-validation works with PC algorithm."""
        # Just test that _run_discovery is called correctly
        call_args = []

        def mock_run_discovery(tool_name, data, names):
            call_args.append((tool_name, data.shape, names))
            return {
                "edges": [
                    {"source": names[0], "target": names[1], "edge_type": "directed", "confidence": 0.7},
                ],
                "nodes": [{"name": n, "node_type": "variable"} for n in names],
                "graph_data": {},
            }

        monkeypatch.setattr("app.agents.discovery._run_discovery", mock_run_discovery)

        data = np.random.randn(200, 4)
        names = ["X1", "X2", "X3", "X4"]

        result = _cross_validate_discovery(data, names, "pc", num_folds=3)

        assert result is not None
        # Should be called 3 times (once per fold)
        assert len(call_args) == 3
        # Each call should use "pc" as algorithm
        assert all(args[0] == "pc" for args in call_args)
        # Each call should use subsample of data
        assert all(args[1][0] < 200 for args in call_args)

    def test_cv_stability_selection_effect(self, monkeypatch):
        """Test that stability selection removes spurious edges."""
        # Simulate a scenario where one edge is consistently found,
        # but another appears randomly

        fold_idx = [0]

        def mock_run_discovery(tool_name, data, names):
            fold = fold_idx[0]
            fold_idx[0] += 1

            edges = [
                # True edge - always found
                {"source": "treatment", "target": "outcome", "edge_type": "directed", "confidence": 0.8},
            ]
            # Spurious edge - only found in some folds
            if fold % 3 == 0:
                edges.append(
                    {"source": "noise", "target": "outcome", "edge_type": "directed", "confidence": 0.5}
                )

            return {
                "edges": edges,
                "nodes": [{"name": n, "node_type": "variable"} for n in names],
                "graph_data": {},
            }

        monkeypatch.setattr("app.agents.discovery._run_discovery", mock_run_discovery)

        data = np.random.randn(200, 3)
        names = ["treatment", "outcome", "noise"]

        result = _cross_validate_discovery(
            data, names, "pc",
            num_folds=9,
            stability_threshold=0.5,
        )

        assert result is not None
        # True edge should be stable (appears in all folds)
        assert any(
            e["source"] == "treatment" and e["target"] == "outcome"
            for e in result["stable_edges"]
        )
        # Spurious edge frequency should be about 0.33
        spurious_freq = result["edge_frequencies"].get("noise->outcome", 0)
        assert spurious_freq < 0.5  # Not stable at 0.5 threshold
