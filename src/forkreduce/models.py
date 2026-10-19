"""OpenAI-compatible models.

forkreduce itself accepts any pydantic-ai ``Model``. This helper is only for the
common case of an OpenAI-style chat endpoint. Retries stay in forkreduce:
the OpenAI client is constructed with ``max_retries=0``.
"""

from __future__ import annotations

import os

from pydantic_ai.models.openai import OpenAIChatModel


def openai_compatible_model(
    model_name: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAIChatModel:
    """Build an ``OpenAIChatModel`` from arguments or environment variables.

    Environment variables, when the matching argument is omitted:

    * ``OPENAI_MODEL`` (or ``FORKREDUCE_MODEL``) for the model name
    * ``OPENAI_API_KEY``
    * ``OPENAI_BASE_URL``
    """
    from openai import AsyncOpenAI
    from pydantic_ai.providers.openai import OpenAIProvider

    from forkreduce.errors import ForkReduceError

    name = model_name or os.environ.get("OPENAI_MODEL") or os.environ.get("FORKREDUCE_MODEL")
    if not name:
        raise ForkReduceError("Pass model_name or set OPENAI_MODEL")
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
    # A local OpenAI-compatible server often has no key. The client still wants
    # a non-empty string when base_url is set.
    resolved_key = key or ("api-key-not-set" if url else None)
    if url and resolved_key is not None:
        client = AsyncOpenAI(api_key=resolved_key, base_url=url, max_retries=0)
    elif resolved_key is not None:
        client = AsyncOpenAI(api_key=resolved_key, max_retries=0)
    else:
        client = AsyncOpenAI(max_retries=0)
    return OpenAIChatModel(name, provider=OpenAIProvider(openai_client=client))
