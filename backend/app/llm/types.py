"""Shared types for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""

    text: str
    raw: Any | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMProvider(Protocol):
    """Interface for LLM providers."""

    name: str

    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return a full completion for the prompt."""

    async def structured_output(self, prompt: str, **kwargs: Any) -> dict[str, Any] | None:
        """Return structured JSON output for the prompt."""

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream response tokens for the prompt."""
