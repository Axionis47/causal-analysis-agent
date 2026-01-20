"""Tests for SQLAlchemy models."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Analysis,
    AnalysisStatus,
    CausalGraph,
    Dataset,
    DiscoveryMethod,
    TreatmentEffect,
    TreatmentMethod,
)


class TestAnalysisModel:
    """Tests for Analysis model."""

    @pytest.mark.asyncio
    async def test_create_analysis(
        self, db_session: AsyncSession, sample_analysis_data: dict
    ):
        """Test creating an analysis."""
        analysis = Analysis(**sample_analysis_data)
        db_session.add(analysis)
        await db_session.commit()
        await db_session.refresh(analysis)

        assert analysis.id is not None
        assert isinstance(analysis.id, uuid.UUID)
        assert analysis.kaggle_url == sample_analysis_data["kaggle_url"]
        assert analysis.status == AnalysisStatus.PENDING
        assert analysis.created_at is not None
        assert analysis.updated_at is not None

    @pytest.mark.asyncio
    async def test_analysis_status_enum(
        self, db_session: AsyncSession, sample_analysis_data: dict
    ):
        """Test analysis status transitions."""
        analysis = Analysis(**sample_analysis_data)
        db_session.add(analysis)
        await db_session.commit()

        # Update status
        analysis.status = AnalysisStatus.RUNNING
        await db_session.commit()
        await db_session.refresh(analysis)

        assert analysis.status == AnalysisStatus.RUNNING

    @pytest.mark.asyncio
    async def test_analysis_config_jsonb(
        self, db_session: AsyncSession, sample_analysis_data: dict
    ):
        """Test JSONB config field."""
        sample_analysis_data["config"] = {
            "treatment": "var_a",
            "outcome": "var_b",
            "confounders": ["var_c", "var_d"],
            "nested": {"key": "value"},
        }
        analysis = Analysis(**sample_analysis_data)
        db_session.add(analysis)
        await db_session.commit()
        await db_session.refresh(analysis)

        assert analysis.config["treatment"] == "var_a"
        assert analysis.config["nested"]["key"] == "value"


class TestDatasetModel:
    """Tests for Dataset model."""

    @pytest.mark.asyncio
    async def test_create_dataset_with_analysis(
        self,
        db_session: AsyncSession,
        sample_analysis_data: dict,
        sample_dataset_data: dict,
    ):
        """Test creating a dataset linked to an analysis."""
        analysis = Analysis(**sample_analysis_data)
        db_session.add(analysis)
        await db_session.commit()

        sample_dataset_data["analysis_id"] = analysis.id
        dataset = Dataset(**sample_dataset_data)
        db_session.add(dataset)
        await db_session.commit()
        await db_session.refresh(dataset)

        assert dataset.id is not None
        assert dataset.analysis_id == analysis.id
        assert len(dataset.files) == 1
        assert dataset.files[0]["name"] == "data.csv"


class TestCausalGraphModel:
    """Tests for CausalGraph model."""

    @pytest.mark.asyncio
    async def test_create_causal_graph(
        self,
        db_session: AsyncSession,
        sample_analysis_data: dict,
        sample_causal_graph_data: dict,
    ):
        """Test creating a causal graph."""
        analysis = Analysis(**sample_analysis_data)
        db_session.add(analysis)
        await db_session.commit()

        sample_causal_graph_data["analysis_id"] = analysis.id
        graph = CausalGraph(**sample_causal_graph_data)
        db_session.add(graph)
        await db_session.commit()
        await db_session.refresh(graph)

        assert graph.id is not None
        assert graph.method == DiscoveryMethod.PC
        assert graph.confidence == 0.85
        assert len(graph.edges) == 1


class TestTreatmentEffectModel:
    """Tests for TreatmentEffect model."""

    @pytest.mark.asyncio
    async def test_create_treatment_effect(
        self,
        db_session: AsyncSession,
        sample_analysis_data: dict,
        sample_treatment_effect_data: dict,
    ):
        """Test creating a treatment effect."""
        analysis = Analysis(**sample_analysis_data)
        db_session.add(analysis)
        await db_session.commit()

        sample_treatment_effect_data["analysis_id"] = analysis.id
        effect = TreatmentEffect(**sample_treatment_effect_data)
        db_session.add(effect)
        await db_session.commit()
        await db_session.refresh(effect)

        assert effect.id is not None
        assert effect.ate == 0.25
        assert effect.method == TreatmentMethod.PROPENSITY_MATCHING

