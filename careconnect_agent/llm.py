"""LLM adapter wrapping the OpenAI SDK with a test-friendly interface."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

from .config import get_settings


logger = logging.getLogger("careconnect_agent.llm")


class LLMError(RuntimeError):
    pass


class LLMClientProtocol(Protocol):
    def chat(self,
             messages: list[dict[str, str]],
             response_format: Literal["text", "json"] = "text",
             temperature: float = 0.0) -> str: ...


@dataclass
class MockLLMClient:
    """Inject canned responses for tests or offline mode.

    ``responses`` is consumed in FIFO order. When exhausted, ``fallback`` is
    used if provided, otherwise an error is raised.
    """
    responses: list[str] = field(default_factory=list)
    fallback: Optional[str] = None

    def chat(self,
             messages: list[dict[str, str]],
             response_format: Literal["text", "json"] = "text",
             temperature: float = 0.0) -> str:
        if self.responses:
            return self.responses.pop(0)
        if self.fallback is not None:
            return self.fallback
        raise LLMError("MockLLMClient has no responses left and no fallback.")


@dataclass
class OpenAILLMClient:
    """Thin wrapper around ``openai.chat.completions.create``.

    JSON mode uses the SDK ``response_format`` parameter plus a one-time retry
    when JSON parsing fails. Any env without an API key should use
    ``MockLLMClient`` instead.
    """

    model: Optional[str] = None
    api_key: Optional[str] = None
    _cached_client: Any = None

    def _get_client(self):
        if self._cached_client is not None:
            return self._cached_client
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - import error only if deps missing
            raise LLMError("openai package is not installed. Run pip install -r requirements.txt.") from e
        settings = get_settings()
        api_key = self.api_key or settings.openai_api_key
        model = self.model or settings.openai_model
        if not api_key:
            raise LLMError("OPENAI_API_KEY missing. Set it in the environment or use MockLLMClient for tests.")
        self._cached_client = OpenAI(api_key=api_key)
        self.model = model
        return self._cached_client

    def chat(self,
             messages: list[dict[str, str]],
             response_format: Literal["text", "json"] = "text",
             temperature: float = 0.0) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        attempts = 2
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                completion = client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content or ""
                if response_format == "json":
                    json.loads(content)  # validate; raise on malformed -> retry
                return content
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning("JSON parse failed on attempt %d/%d: %s", attempt + 1, attempts, e)
            except Exception as e:  # pragma: no cover - network errors
                last_error = e
                logger.warning("LLM call failed on attempt %d/%d: %s", attempt + 1, attempts, e)
        raise LLMError(f"LLM chat failed after {attempts} attempts") from last_error


def build_default_client() -> LLMClientProtocol:
    """Return the standard client based on env, or a Mock client if no LLM key."""
    settings = get_settings()
    if settings.llm_available:
        return OpenAILLMClient()
    return MockLLMClient(fallback="(LLM unavailable)")
