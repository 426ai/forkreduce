"""Opt-in provider construction and response-level (not history-level) accounting.

Constructors do not perform inference. The CLI owns approval and spend gates.
Every method gets the same provider settings, SDK retries disabled, and automatic
provider caching (OpenAI) or identical explicit five-minute breakpoints (Anthropic).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any, Literal, cast

from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.settings import ModelSettings

from benchmarks.schema import MethodResult

CACHE_POLICY = {
    "openai": "provider_automatic",
    "anthropic": "instructions_tools_messages_5m",
}

BENCHMARK_USER_AGENT = "forkreduce-bench/cuad-smoke"


def provider_model(
    provider: str,
    model_name: str | None = None,
    *,
    api: Literal["chat", "responses"] = "chat",
    settings: Mapping[str, Any] | None = None,
    profile: ModelProfileSpec | None = None,
) -> Model:
    """Build a provider using environment credentials, without making requests.

    ``settings`` uses Pydantic AI names; ``prompt_cache_key`` is translated to
    ``openai_prompt_cache_key`` for either OpenAI API. SDK retries stay disabled
    even when supplied settings request otherwise.
    """
    if api not in ("chat", "responses"):
        raise ValueError("unknown_provider_api")
    chosen = dict(settings or {})
    if "prompt_cache_key" in chosen:
        chosen["openai_prompt_cache_key"] = chosen.pop("prompt_cache_key")
    headers = dict(chosen.get("extra_headers") or {})
    for name in tuple(headers):
        if name.lower() == "user-agent" and name != "User-Agent":
            headers["User-Agent"] = headers.pop(name)
    headers.setdefault("User-Agent", BENCHMARK_USER_AGENT)
    chosen["extra_headers"] = headers
    chosen["max_retries"] = 0
    if provider == "openai":
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        name = model_name or os.environ.get("OPENAI_MODEL") or os.environ.get("FORKREDUCE_MODEL")
        key = os.environ.get("OPENAI_API_KEY")
        url = os.environ.get("OPENAI_BASE_URL")
        if not name:
            raise ValueError("openai_model_missing")
        if not (key or url):
            raise ValueError("openai_credentials_missing")
        client_options: dict[str, Any] = {
            "api_key": key or "api-key-not-set",
            "max_retries": 0,
            "default_headers": {"User-Agent": BENCHMARK_USER_AGENT},
        }
        if url:
            client_options["base_url"] = url
        if chosen.get("timeout") is not None:
            client_options["timeout"] = chosen["timeout"]
        model_class = OpenAIResponsesModel if api == "responses" else OpenAIChatModel
        return model_class(
            name,
            provider=OpenAIProvider(openai_client=AsyncOpenAI(**client_options)),
            profile=profile,
            settings=cast(ModelSettings, chosen),
        )
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        if api != "chat":
            raise ValueError("anthropic_responses_unsupported")
        name = model_name or os.environ.get("ANTHROPIC_MODEL")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not name:
            raise ValueError("anthropic_model_missing")
        if not key:
            raise ValueError("anthropic_credentials_missing")
        defaults = {
            "anthropic_cache_instructions": "5m",
            "anthropic_cache_tool_definitions": "5m",
            "anthropic_cache_messages": "5m",
        }
        defaults.update(chosen)
        return AnthropicModel(
            name,
            provider=AnthropicProvider(
                anthropic_client=AsyncAnthropic(
                    api_key=key,
                    max_retries=0,
                    default_headers={"User-Agent": BENCHMARK_USER_AGENT},
                )
            ),
            settings=cast(ModelSettings, defaults),
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
        "reasoning_tokens",
    }
    details = {
        key: value
        for key, value in raw.items()
        if key in allowed and isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }
    for name in ("prompt_tokens_details", "input_tokens_details"):
        nested = raw.get(name)
        if isinstance(nested, Mapping):
            value = nested.get("cached_tokens")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                details.setdefault("cached_tokens", value)
    for name in ("completion_tokens_details", "output_tokens_details"):
        nested = raw.get(name)
        if isinstance(nested, Mapping):
            value = nested.get("reasoning_tokens")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                details.setdefault("reasoning_tokens", value)

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
        "reasoning_tokens": None if missing else count("reasoning_tokens", "reasoning_tokens"),
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
        except Exception as exc:
            self.result.metadata["requests_without_response"] = (
                self.result.metadata.get("requests_without_response", 0) + 1
            )
            # Provider bodies may reflect prompts or secrets. Retain only known
            # parameter names for diagnosing compatibility failures.
            body = str(getattr(exc, "body", "")).lower()
            hints = [
                name
                for name in (
                    "tool_choice",
                    "thinking",
                    "strict",
                    "max_tokens",
                    "max_completion_tokens",
                    "temperature",
                    "model_not_found",
                    "schema",
                    "reasoning_content",
                )
                if name in body
            ]
            if hints:
                self.result.metadata.setdefault("provider_error_parameters", []).append(hints)
            raise
        # Account before any schema validation, output validators, tools, or retries.
        self.result.usage.append(usage_record(response, stage=self.stage.get()))
        return response
