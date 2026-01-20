"""Tool registry with feature flags and performance tracking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings

logger = logging.getLogger(__name__)


class ToolStatus:
    """Tool status enumeration."""

    STABLE = "stable"
    BETA = "beta"
    ALPHA = "alpha"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


@dataclass
class ToolMetadata:
    name: str
    version: str
    status: str
    min_sample_size: int
    supported_data_types: list[str]
    performance_tier: str
    tool_type: str
    metrics: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry for causal inference tools with filtering."""

    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = Path(config_path or settings.TOOL_CONFIG_PATH)
        self._tools: dict[str, ToolMetadata] = {}
        self._load_tools()

    def _load_tools(self) -> None:
        if not self._config_path.exists():
            logger.warning("Tool config not found at %s", self._config_path)
            return
        payload = yaml.safe_load(self._config_path.read_text()) or {}
        for tool_type, tools in payload.items():
            for name, meta in (tools or {}).items():
                self.register_tool(tool_type, meta)

    def register_tool(self, tool_type: str, meta: dict[str, Any]) -> None:
        tool = ToolMetadata(
            name=meta["name"],
            version=str(meta.get("version", "1.0")),
            status=meta.get("status", ToolStatus.STABLE),
            min_sample_size=int(meta.get("min_sample_size", 0)),
            supported_data_types=list(meta.get("supported_data_types", [])),
            performance_tier=meta.get("performance_tier", "standard"),
            tool_type=tool_type,
        )
        self._tools[f"{tool_type}:{tool.name}"] = tool

    def available_tools(
        self,
        tool_type: str,
        *,
        sample_size: int,
        data_types: list[str],
        overrides: dict[str, Any] | None = None,
    ) -> list[ToolMetadata]:
        overrides = overrides or {}
        results: list[ToolMetadata] = []
        for key, tool in self._tools.items():
            if tool.tool_type != tool_type:
                continue
            if not self._is_enabled(tool, overrides=overrides):
                continue
            if sample_size < tool.min_sample_size:
                continue
            if tool.supported_data_types:
                if not set(data_types).intersection(tool.supported_data_types):
                    continue
            results.append(tool)
        return results

    def track_performance(self, tool_name: str, success: bool, elapsed: float) -> None:
        tool = self._tools.get(tool_name)
        if not tool:
            return
        metrics = tool.metrics
        metrics.setdefault("runs", 0)
        metrics.setdefault("successes", 0)
        metrics.setdefault("failures", 0)
        metrics.setdefault("total_time", 0.0)
        metrics["runs"] += 1
        metrics["successes"] += 1 if success else 0
        metrics["failures"] += 0 if success else 1
        metrics["total_time"] += elapsed
        metrics["last_run_at"] = time.time()

    def _is_enabled(self, tool: ToolMetadata, overrides: dict[str, Any]) -> bool:
        if tool.status == ToolStatus.DISABLED:
            return False
        if tool.status == ToolStatus.DEPRECATED:
            return overrides.get("allow_deprecated", False)
        if tool.status == ToolStatus.ALPHA and not settings.ENABLE_ALPHA_TOOLS:
            return False
        if tool.status == ToolStatus.BETA and not settings.ENABLE_BETA_TOOLS:
            if tool.name not in {"ges", "fci", "doubly_robust", "instrumental_variable"}:
                return False
        if tool.name == "ges" and not settings.ENABLE_GES:
            return False
        if tool.name == "fci" and not settings.ENABLE_FCI:
            return False
        if tool.name == "doubly_robust" and not settings.ENABLE_DOUBLY_ROBUST:
            return False
        if tool.name == "instrumental_variable" and not settings.ENABLE_IV:
            return False
        return True
