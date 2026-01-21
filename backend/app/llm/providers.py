"""Provider implementations for LLM abstraction layer."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from app.core.config import settings
from app.llm.types import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class BaseProvider(LLMProvider):
    """Base provider helpers for JSON parsing."""

    name: str = "base"

    async def structured_output(self, prompt: str, **kwargs: Any) -> dict[str, Any] | None:
        response = await self.complete(prompt, **kwargs)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            logger.warning("%s returned non-JSON response", self.name)
            return None

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        response = await self.complete(prompt, **kwargs)
        yield response.text


class VertexAIProvider(BaseProvider):
    """Vertex AI Gemini provider."""

    name = "vertex"

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        self._model_name = model_name
        self._models: dict[str, Any] = {}

    def _get_model(self, model_name: str | None = None):
        if not settings.VERTEX_AI_PROJECT:
            raise ValueError("VERTEX_AI_PROJECT is not configured")
        try:
            from vertexai import init
            from vertexai.generative_models import GenerativeModel
        except ImportError as exc:
            raise RuntimeError("vertexai SDK not installed") from exc

        selected = model_name or self._model_name
        if selected not in self._models:
            init(project=settings.VERTEX_AI_PROJECT, location=settings.VERTEX_AI_LOCATION)
            self._models[selected] = GenerativeModel(selected)
        return self._models[selected]

    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        model_name = kwargs.pop("model", self._model_name)
        model = self._get_model(model_name)
        response = await model.generate_content_async(prompt)
        text = response.text or ""
        token_usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            raw=response,
            prompt_tokens=getattr(token_usage, "prompt_token_count", None),
            completion_tokens=getattr(token_usage, "candidates_token_count", None),
            total_tokens=getattr(token_usage, "total_token_count", None),
        )


class OpenAIProvider(BaseProvider):
    """OpenAI chat completion provider."""

    name = "openai"

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self._model_name = model_name
        self._client = None

    def _get_client(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        model_name = kwargs.pop("model", self._model_name)
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.2),
        )
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            raw=response,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider."""

    name = "anthropic"

    def __init__(self, model_name: str = "claude-3-5-sonnet-20240620") -> None:
        self._model_name = model_name
        self._client = None

    def _get_client(self):
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        return self._client

    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        model_name = kwargs.pop("model", self._model_name)
        response = await client.messages.create(
            model=model_name,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.2),
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        usage = response.usage
        return LLMResponse(
            text=text,
            raw=response,
            prompt_tokens=usage.input_tokens if usage else None,
            completion_tokens=usage.output_tokens if usage else None,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else None,
        )


class MockProvider(BaseProvider):
    """Mock provider for testing without real API keys."""

    name = "mock"

    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        import json
        # Generate reasonable mock responses based on prompt content
        if "executive summary" in prompt.lower():
            response = json.dumps({
                "overview": "This analysis examines causal relationships in the dataset.",
                "key_findings": ["Treatment effect detected", "Moderate confidence in results"],
                "recommendations": ["Collect more data", "Validate assumptions"]
            })
        elif "select" in prompt.lower() and "file" in prompt.lower():
            response = json.dumps({"selected_file": "train.csv", "reasoning": "Main training dataset"})
        elif "causal" in prompt.lower() and "graph" in prompt.lower():
            response = json.dumps({"edges": [], "method": "PC", "confidence": 0.7})
        elif "treatment" in prompt.lower() or "effect" in prompt.lower():
            response = json.dumps({
                "treatment": "Pclass",
                "outcome": "Survived",
                "ate": -0.15,
                "confidence_interval": [-0.2, -0.1],
                "method": "propensity_score"
            })
        else:
            response = json.dumps({"result": "mock_response", "status": "success"})

        return LLMResponse(
            text=response,
            raw={"mock": True},
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response.split()),
            total_tokens=len(prompt.split()) + len(response.split()),
        )
