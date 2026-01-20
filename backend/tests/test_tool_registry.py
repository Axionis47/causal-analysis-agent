"""Tests for tool registry feature flags."""

from app.tools.registry import ToolRegistry
from app.core.config import settings


def test_tool_registry_filters_beta_tools(monkeypatch):
    registry = ToolRegistry()
    monkeypatch.setattr(settings, "ENABLE_BETA_TOOLS", False)
    tools = registry.available_tools("discovery", sample_size=1000, data_types=["numerical"], overrides={})
    tool_names = [tool.name for tool in tools]
    assert "pc" in tool_names
    assert "ges" not in tool_names


def test_tool_registry_allows_beta_tools(monkeypatch):
    registry = ToolRegistry()
    monkeypatch.setattr(settings, "ENABLE_BETA_TOOLS", True)
    monkeypatch.setattr(settings, "ENABLE_GES", True)
    tools = registry.available_tools("discovery", sample_size=1000, data_types=["numerical"], overrides={})
    tool_names = [tool.name for tool in tools]
    assert "ges" in tool_names
