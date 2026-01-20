"""Tests for cost-based routing helpers."""

from __future__ import annotations

from app.llm.cost_router import PromptComplexity, classify_prompt_complexity, select_model_for_complexity


def test_classify_prompt_complexity_simple():
    assert classify_prompt_complexity("Hello world") == PromptComplexity.SIMPLE


def test_classify_prompt_complexity_keyword():
    assert classify_prompt_complexity("Please analyze this") == PromptComplexity.COMPLEX


def test_classify_prompt_complexity_length():
    prompt = "a" * 600
    assert classify_prompt_complexity(prompt) == PromptComplexity.COMPLEX


def test_select_model_for_complexity():
    assert select_model_for_complexity(PromptComplexity.SIMPLE, "openai") == "gpt-3.5-turbo"
    assert select_model_for_complexity(PromptComplexity.SIMPLE, "anthropic") == "claude-3-haiku"
    assert select_model_for_complexity(PromptComplexity.SIMPLE, "vertex") == "gemini-1.0-pro"
    assert select_model_for_complexity(PromptComplexity.COMPLEX, "openai") == "gpt-4o"
    assert select_model_for_complexity(PromptComplexity.COMPLEX, "anthropic") == "claude-3-5-sonnet"
    assert select_model_for_complexity(PromptComplexity.COMPLEX, "vertex") == "gemini-1.5-pro"
