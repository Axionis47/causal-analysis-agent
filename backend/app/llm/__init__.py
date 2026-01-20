"""LLM abstraction layer with provider routing and fallback."""

from app.llm.router import LLMRouter

__all__ = ["LLMRouter"]
