"""Cost-based routing helpers for LLM model selection."""

from __future__ import annotations

import enum

from app.core.config import settings


class PromptComplexity(str, enum.Enum):
    """Prompt complexity classification."""

    SIMPLE = "simple"
    COMPLEX = "complex"


_COMPLEX_KEYWORDS = ("analyze", "explain", "reason", "compare")

_SIMPLE_MODEL_MAP = {
    "vertex": "gemini-1.0-pro",
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-3-haiku",
}

_COMPLEX_MODEL_MAP = {
    "vertex": "gemini-1.5-pro",
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet",
}


def classify_prompt_complexity(prompt: str) -> PromptComplexity:
    """Classify prompt complexity based on heuristics."""
    prompt_lower = prompt.lower()
    if len(prompt) > settings.LLM_SIMPLE_PROMPT_MAX_LENGTH:
        return PromptComplexity.COMPLEX
    if any(keyword in prompt_lower for keyword in _COMPLEX_KEYWORDS):
        return PromptComplexity.COMPLEX
    return PromptComplexity.SIMPLE


def select_model_for_complexity(complexity: PromptComplexity, provider: str) -> str:
    """Select a model name based on complexity and provider."""
    if complexity == PromptComplexity.SIMPLE:
        return _SIMPLE_MODEL_MAP[provider]
    return _COMPLEX_MODEL_MAP[provider]
