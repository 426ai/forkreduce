"""OpenAI-compatible models.

forkreduce itself accepts any pydantic-ai ``Model``. This helper is only for the
common case of an OpenAI-style chat endpoint. Retries stay in forkreduce:
the OpenAI client is constructed with ``max_retries=0``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import httpx
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from pydantic_ai.models.openai import OpenAIChatModel

# These change how the HTTP call is made. They are not part of the JSON body,
# so they stay out of the Temporal execution fingerprint.
_TRANSPORT_SETTING_KEYS = frozenset({"timeout", "extra_headers"})


def openai_compatible_model(
    model_name: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    profile: ModelProfileSpec | None = None,
    extra_body: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    stop_sequences: list[str] | None = None,
    parallel_tool_calls: bool | None = None,
    extra_headers: dict[str, str] | None = None,
    settings: ModelSettings | None = None,
) -> OpenAIChatModel:
    """Build an ``OpenAIChatModel`` from arguments or environment variables.

    Environment variables, when the matching argument is omitted:

    * ``OPENAI_MODEL`` (or ``FORKREDUCE_MODEL``) for the model name
    * ``OPENAI_API_KEY``
    * ``OPENAI_BASE_URL``

    ``profile`` is a pydantic-ai ``ModelProfile`` (or a callable that returns
    one). ``extra_body`` is merged into the chat-completions JSON body, which
    is how many OpenAI-compatible servers turn provider "thinking" off, for
    example ``{"chat_template_kwargs": {"enable_thinking": False}}``.

    ``settings`` accepts any other pydantic-ai ``ModelSettings`` key, including
    OpenAI-specific ones. An explicit argument overrides the same key in
    ``settings``. ``timeout`` is applied both as a model setting and as the
    OpenAI client's default. Sampling keys, ``extra_body``, and ``profile``
    can change what the provider sees; see the README for which of those bust
    a prefix cache.
    """
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:
        raise ImportError("OpenAI support requires `pip install 'forkreduce[openai]'` or 'forkreduce[full]'.") from exc

    from forkreduce.errors import ForkReduceError

    name = model_name or os.environ.get("OPENAI_MODEL") or os.environ.get("FORKREDUCE_MODEL")
    if not name:
        raise ForkReduceError("Pass model_name or set OPENAI_MODEL")
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
    # A local OpenAI-compatible server often has no key. The client still wants
    # a non-empty string when base_url is set.
    resolved_key = key or ("api-key-not-set" if url else None)
    chosen = _model_settings(
        settings,
        timeout=timeout,
        extra_body=extra_body,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop_sequences=stop_sequences,
        parallel_tool_calls=parallel_tool_calls,
        extra_headers=extra_headers,
    )
    client_timeout = None if chosen is None else chosen.get("timeout")
    client = _async_client(resolved_key, url, client_timeout)
    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(openai_client=client),
        profile=profile,
        settings=chosen,
    )


def request_body_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Model settings that are sent in the request body.

    ``timeout`` and ``extra_headers`` are transport options. Everything else
    pydantic-ai puts on ``ModelSettings`` can change the JSON body, including
    sampling parameters and ``extra_body``.
    """
    if not settings:
        return {}
    return {str(key): value for key, value in settings.items() if key not in _TRANSPORT_SETTING_KEYS}


def canonical_value(value: Any) -> Any:
    """A JSON-ready copy whose sets do not depend on hash order.

    Classes and other callables become ``module.qualname``. Used for the
    Temporal execution fingerprint, which must be stable for identical
    profiles and settings.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((canonical_value(item) for item in value), key=_canonical_sort_key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonical_value(item) for item in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", "")
        qualname = getattr(value, "__qualname__", None)
        if isinstance(qualname, str):
            return f"{module}.{qualname}"
        return repr(value)
    return str(value)


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _model_settings(
    settings: ModelSettings | None,
    **explicit: Any,
) -> ModelSettings | None:
    merged: dict[str, Any] = dict(settings or {})
    for key, value in explicit.items():
        if value is not None:
            merged[key] = value
    if not merged:
        return None
    return cast(ModelSettings, merged)


def _async_client(api_key: str | None, base_url: str | None, timeout: Any) -> Any:
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {"max_retries": 0}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    return AsyncOpenAI(**kwargs)
