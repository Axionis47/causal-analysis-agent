"""Prompt regression checks for agent templates."""

from app.prompts.discovery import TEMPLATE as DISCOVERY_TEMPLATE
from app.prompts.eda import TEMPLATE as EDA_TEMPLATE
from app.prompts.reporting import TEMPLATE as REPORT_TEMPLATE


def test_eda_prompt_has_required_sections():
    assert "Analyze this dataset summary" in EDA_TEMPLATE
    assert "Summary:" in EDA_TEMPLATE
    assert "Return JSON" in EDA_TEMPLATE


def test_discovery_prompt_has_required_sections():
    assert "Interpret the causal graph" in DISCOVERY_TEMPLATE
    assert "Nodes:" in DISCOVERY_TEMPLATE
    assert "Edges:" in DISCOVERY_TEMPLATE


def test_report_prompt_has_required_sections():
    assert "executive summary" in REPORT_TEMPLATE.lower()
    assert "overview" in REPORT_TEMPLATE
