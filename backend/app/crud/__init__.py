"""CRUD utilities for database operations."""

from app.crud.base import CRUDBase
from app.crud.analysis import CRUDAnalysis, analysis_crud
from app.crud.analysis_version import CRUDAnalysisVersion, analysis_version_crud
from app.crud.dataset import CRUDDataset, dataset_crud

__all__ = [
    "CRUDBase",
    "CRUDAnalysis",
    "analysis_crud",
    "CRUDAnalysisVersion",
    "analysis_version_crud",
    "CRUDDataset",
    "dataset_crud",
]

