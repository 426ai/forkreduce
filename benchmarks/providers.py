"""Opt-in provider construction and response-level (not history-level) accounting.

Constructors do not perform inference. The CLI owns approval and spend gates.
Every method gets the same provider settings, SDK retries disabled, and automatic
provider caching (OpenAI) or identical explicit five-minute breakpoints (Anthropic).
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel

from benchmarks.schema import MethodResult

CACHE_POLICY = {
    "openai": "provider_automatic",
    "anthropic": "instructions_tools_messages_5m",
}


def provider_model(provider: str, model_name: str | None = None) -> Model:
    """Build a provider only after the caller's approval gate has succeeded."""
    if provider == "openai":
        from forkreduce.models import openai_compatible_model

        name = model_name or os.environ.get("OPENAI_MODEL") or os.environ.get("FORKREDUCE_MODEL")
        if not name:
            raise ValueError("openai_model_missing")
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL")):
            raise ValueError("openai_credentials_missing")
        return openai_compatible_model(name)
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        name = model_name or os.environ.get("ANTHROPIC_MODEL")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not name:
            raise ValueError("anthropic_model_missing")
        if not key:
            raise ValueError("anthropic_credentials_missing")
        return AnthropicModel(
            name,
            provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key=key, max_retries=0)),
            settings={
                "anthropic_cache_instructions": "5m",
                "anthropic_cache_tool_definitions": "5m",
                "anthropic_cache_messages": "5m",
            },
        )
    raise ValueError("unknown_provider")


def usage_record(response: Any, *, stage: str) -> dict[str, Any]:
    """Keep unknown usage unknown; zero is reserved for observed provider zeros."""
    usage = getattr(response, "usage", None)
    raw = getattr(usage, "details", None) or {}
    allowed = {
        "prompt_tokens",
        "completion_tokens",
        "prompt_cache_hit_tokens",
        "cached_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
        "input_tokens",
        "output_tokens",
    }
    details = {
        key: value
        for key, value in raw.items()
        if key in allowed and isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }
    nested = raw.get("prompt_tokens_details")
    if isinstance(nested, dict) and isinstance(nested.get("cached_tokens"), int):
        value = nested["cached_tokens"]
        if not isinstance(value, bool) and value >= 0:
            details.setdefault("cached_tokens", value)

    def count(name: str, *fallbacks: str) -> int | None:
        value = getattr(usage, name, None)
        if not value:
            value = next((details[key] for key in fallbacks if key in details), value)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    input_tokens = count("input_tokens", "prompt_tokens")
    output_tokens = count("output_tokens", "completion_tokens")
    # RequestUsage defaults to all zeros even when a gateway omitted usage entirely.
    # A nonempty details mapping or an observed nonzero count is evidence of usage.
    missing = usage is None or not (input_tokens or output_tokens or details)
    return {
        "input_tokens": None if missing else input_tokens,
        "output_tokens": None if missing else output_tokens,
        "cache_read_tokens": None
        if missing
        else count(
            "cache_read_tokens",
            "prompt_cache_hit_tokens",
            "cached_tokens",
            "cache_read_input_tokens",
            "cache_read_tokens",
        ),
        "cache_write_tokens": None
        if missing
        else count("cache_write_tokens", "cache_creation_input_tokens", "cache_write_tokens"),
        "usage_missing": missing or input_tokens is None or output_tokens is None,
        "details": details,
        "stage": stage,
    }


class RecordingModel(WrapperModel):
    """One private recorder per run; ContextVar keeps concurrent fork stages apart."""

    def __init__(self, wrapped: Model, result: MethodResult) -> None:
        super().__init__(wrapped)
        self.result = result
        self.stage: ContextVar[str] = ContextVar("benchmark_stage", default="extraction")

    async def request(self, messages: Any, model_settings: Any, model_request_parameters: Any) -> Any:
        try:
            response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        except Exception:
            self.result.metadata["requests_without_response"] = (
                self.result.metadata.get("requests_without_response", 0) + 1
            )
            raise
        # Account before any schema validation, output validators, tools, or retries.
        self.result.usage.append(usage_record(response, stage=self.stage.get()))
        return response
