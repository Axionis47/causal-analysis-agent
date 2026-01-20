"""Tests for analysis versioning functionality."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.analysis import analysis_crud
from app.crud.analysis_version import analysis_version_crud, compute_config_hash
from app.models.analysis import Analysis, AnalysisStatus
from app.models.analysis_version import AnalysisVersion
from app.services.version_comparison import (
    compute_config_diff,
    compute_version_similarity,
    generate_diff_summary,
)


class TestComputeConfigHash:
    """Tests for config hash computation."""

    def test_hash_is_deterministic(self):
        """Same config should produce same hash."""
        config = {"key": "value", "nested": {"a": 1, "b": 2}}
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        assert hash1 == hash2

    def test_hash_is_64_characters(self):
        """SHA-256 hash should be 64 hex characters."""
        config = {"key": "value"}
        hash_value = compute_config_hash(config)
        assert len(hash_value) == 64

    def test_hash_differs_for_different_configs(self):
        """Different configs should produce different hashes."""
        config1 = {"key": "value1"}
        config2 = {"key": "value2"}
        assert compute_config_hash(config1) != compute_config_hash(config2)

    def test_hash_ignores_key_order(self):
        """Hash should be same regardless of key order."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        assert compute_config_hash(config1) == compute_config_hash(config2)


class TestConfigDiff:
    """Tests for config diff computation."""

    def test_empty_configs(self):
        """Diff of empty configs should show no changes."""
        diff = compute_config_diff({}, {})
        assert diff["added"] == {}
        assert diff["removed"] == {}
        assert diff["modified"] == {}
        assert diff["unchanged"] == {}

    def test_added_fields(self):
        """New fields in config2 should be in added."""
        config1 = {"existing": 1}
        config2 = {"existing": 1, "new": 2}
        diff = compute_config_diff(config1, config2)
        assert "new" in diff["added"]
        assert diff["added"]["new"] == 2

    def test_removed_fields(self):
        """Fields missing from config2 should be in removed."""
        config1 = {"existing": 1, "old": 2}
        config2 = {"existing": 1}
        diff = compute_config_diff(config1, config2)
        assert "old" in diff["removed"]
        assert diff["removed"]["old"] == 2

    def test_modified_fields(self):
        """Changed fields should be in modified with old/new values."""
        config1 = {"key": "old_value"}
        config2 = {"key": "new_value"}
        diff = compute_config_diff(config1, config2)
        assert "key" in diff["modified"]
        assert diff["modified"]["key"]["old"] == "old_value"
        assert diff["modified"]["key"]["new"] == "new_value"

    def test_unchanged_fields(self):
        """Unchanged fields should be in unchanged."""
        config1 = {"key": "value"}
        config2 = {"key": "value"}
        diff = compute_config_diff(config1, config2)
        assert "key" in diff["unchanged"]

    def test_nested_config_diff(self):
        """Nested config changes should be detected."""
        config1 = {"outer": {"inner": 1}}
        config2 = {"outer": {"inner": 2}}
        diff = compute_config_diff(config1, config2)
        assert "outer.inner" in diff["modified"]


class TestDiffSummary:
    """Tests for diff summary generation."""

    def test_no_changes(self):
        """No changes should return 'No changes'."""
        diff = {"added": {}, "removed": {}, "modified": {}}
        summary = generate_diff_summary(diff)
        assert summary == "No changes"

    def test_single_addition(self):
        """Single addition should use singular form."""
        diff = {"added": {"key": 1}, "removed": {}, "modified": {}}
        summary = generate_diff_summary(diff)
        assert "1 field added" in summary

    def test_multiple_changes(self):
        """Multiple changes should show all counts."""
        diff = {
            "added": {"a": 1, "b": 2},
            "removed": {"c": 3},
            "modified": {"d": {"old": 1, "new": 2}},
        }
        summary = generate_diff_summary(diff)
        assert "2 fields added" in summary
        assert "1 field removed" in summary
        assert "1 field modified" in summary


class TestVersionSimilarity:
    """Tests for version similarity computation."""

    def test_identical_configs(self):
        """Identical configs should have similarity of 1.0."""
        config = {"key": "value"}
        similarity = compute_version_similarity(config, config)
        assert similarity == 1.0

    def test_empty_configs(self):
        """Empty configs should have similarity of 1.0."""
        similarity = compute_version_similarity({}, {})
        assert similarity == 1.0

    def test_completely_different_configs(self):
        """Completely different configs should have low similarity."""
        config1 = {"a": 1, "b": 2}
        config2 = {"c": 3, "d": 4}
        similarity = compute_version_similarity(config1, config2)
        assert similarity < 0.5

    def test_partial_overlap(self):
        """Partial overlap should have moderate similarity."""
        config1 = {"a": 1, "b": 2}
        config2 = {"a": 1, "c": 3}
        similarity = compute_version_similarity(config1, config2)
        assert 0.0 < similarity < 1.0


@pytest.mark.asyncio
class TestVersionCRUD:
    """Tests for AnalysisVersion CRUD operations."""

    @pytest_asyncio.fixture
    async def test_analysis(self, db_session: AsyncSession) -> Analysis:
        """Create a test analysis."""
        analysis = await analysis_crud.create(
            db_session,
            obj_in={
                "kaggle_url": "https://www.kaggle.com/datasets/test/sample",
                "status": AnalysisStatus.PENDING,
                "config": {"treatment": "x", "outcome": "y"},
                "user_id": uuid.uuid4(),
            },
        )
        return analysis

    async def test_create_version(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test creating a new version."""
        version = await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"new": "config"},
            changed_by=test_analysis.user_id,
            change_summary="Test change",
        )

        assert version.analysis_id == test_analysis.id
        assert version.version_number == 1
        assert version.config == {"new": "config"}
        assert version.change_summary == "Test change"
        assert len(version.config_hash) == 64

    async def test_version_auto_increment(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test version numbers auto-increment."""
        v1 = await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"v": 1},
        )
        v2 = await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"v": 2},
        )
        v3 = await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"v": 3},
        )

        assert v1.version_number == 1
        assert v2.version_number == 2
        assert v3.version_number == 3

    async def test_get_version(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test getting a specific version."""
        await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"v": 1},
        )
        await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"v": 2},
        )

        version = await analysis_version_crud.get_version(
            db_session, test_analysis.id, 2
        )
        assert version is not None
        assert version.version_number == 2
        assert version.config == {"v": 2}

    async def test_get_nonexistent_version(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test getting a version that doesn't exist."""
        version = await analysis_version_crud.get_version(
            db_session, test_analysis.id, 999
        )
        assert version is None

    async def test_get_latest_version(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test getting the latest version."""
        await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"v": 1},
        )
        await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"v": 2},
        )

        latest = await analysis_version_crud.get_latest_version(
            db_session, test_analysis.id
        )
        assert latest is not None
        assert latest.version_number == 2

    async def test_get_by_analysis(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test getting all versions for an analysis."""
        for i in range(5):
            await analysis_version_crud.create_version(
                db_session,
                analysis_id=test_analysis.id,
                config={"v": i + 1},
            )

        versions = await analysis_version_crud.get_by_analysis(
            db_session, test_analysis.id
        )
        assert len(versions) == 5
        # Should be ordered by version_number desc
        assert versions[0].version_number == 5
        assert versions[4].version_number == 1

    async def test_get_version_count(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test counting versions."""
        for i in range(3):
            await analysis_version_crud.create_version(
                db_session,
                analysis_id=test_analysis.id,
                config={"v": i + 1},
            )

        count = await analysis_version_crud.get_version_count(
            db_session, test_analysis.id
        )
        assert count == 3


@pytest.mark.asyncio
class TestAnalysisCRUDVersioning:
    """Tests for Analysis CRUD versioning methods."""

    @pytest_asyncio.fixture
    async def test_analysis(self, db_session: AsyncSession) -> Analysis:
        """Create a test analysis."""
        analysis = await analysis_crud.create(
            db_session,
            obj_in={
                "kaggle_url": "https://www.kaggle.com/datasets/test/sample",
                "status": AnalysisStatus.PENDING,
                "config": {"initial": "config"},
                "user_id": uuid.uuid4(),
            },
        )
        return analysis

    async def test_update_with_versioning_creates_version(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test that updating config creates a new version."""
        # Create initial version
        await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config=test_analysis.config,
        )

        # Update with new config
        await analysis_crud.update_with_versioning(
            db_session,
            db_obj=test_analysis,
            obj_in={"config": {"updated": "config"}},
            changed_by=test_analysis.user_id,
            change_summary="Updated config",
        )

        # Should have 2 versions now
        count = await analysis_version_crud.get_version_count(
            db_session, test_analysis.id
        )
        assert count == 2

    async def test_update_with_versioning_skips_unchanged_config(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test that updating with same config doesn't create new version."""
        # Create initial version
        await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config=test_analysis.config,
        )

        # Update with same config
        await analysis_crud.update_with_versioning(
            db_session,
            db_obj=test_analysis,
            obj_in={"config": test_analysis.config},
        )

        # Should still have only 1 version
        count = await analysis_version_crud.get_version_count(
            db_session, test_analysis.id
        )
        assert count == 1

    async def test_create_manual_snapshot(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test creating a manual snapshot."""
        snapshot = await analysis_crud.create_manual_snapshot(
            db_session,
            analysis_id=test_analysis.id,
            changed_by=test_analysis.user_id,
            change_summary="Pre-change snapshot",
        )

        assert snapshot.is_manual_snapshot is True
        assert snapshot.change_summary == "Pre-change snapshot"
        assert snapshot.config == test_analysis.config

    async def test_revert_to_version(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test reverting to a previous version."""
        # Create version 1
        await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"v": 1},
        )

        # Update to create version 2
        test_analysis = await analysis_crud.update(
            db_session,
            db_obj=test_analysis,
            obj_in={"config": {"v": 2}},
        )
        await analysis_version_crud.create_version(
            db_session,
            analysis_id=test_analysis.id,
            config={"v": 2},
        )

        # Revert to version 1
        reverted = await analysis_crud.revert_to_version(
            db_session,
            analysis_id=test_analysis.id,
            version_number=1,
            changed_by=test_analysis.user_id,
        )

        # Analysis config should be from v1
        assert reverted.config == {"v": 1}

        # Should have created a new version (v3)
        count = await analysis_version_crud.get_version_count(
            db_session, test_analysis.id
        )
        assert count == 3

        # The new version should have parent_version_id set
        latest = await analysis_version_crud.get_latest_version(
            db_session, test_analysis.id
        )
        assert latest.parent_version_id is not None

    async def test_revert_to_nonexistent_version_raises(
        self, db_session: AsyncSession, test_analysis: Analysis
    ):
        """Test that reverting to nonexistent version raises error."""
        with pytest.raises(ValueError, match="not found"):
            await analysis_crud.revert_to_version(
                db_session,
                analysis_id=test_analysis.id,
                version_number=999,
            )
