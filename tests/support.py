"""Shared scripted models for offline tests."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

Handler = Callable[
    [list[ModelMessage], AgentInfo, int],
    ModelResponse | Awaitable[ModelResponse],
]


@dataclass
class Scripted:
    """FunctionModel that records calls and the highest concurrency it observed."""

    handler: Handler
    calls: list[tuple[list[ModelMessage], AgentInfo]] = field(default_factory=list)
    max_inflight: int = 0
    _inflight: int = 0

    def model(self) -> FunctionModel:
        scripted = self

        async def _call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            scripted._inflight += 1
            scripted.max_inflight = max(scripted.max_inflight, scripted._inflight)
            scripted.calls.append((messages, info))
            try:
                result = scripted.handler(messages, info, len(scripted.calls))
                if inspect.isawaitable(result):
                    result = await result
                return result
            finally:
                scripted._inflight -= 1

        return FunctionModel(_call)


def usage(
    *,
    input_tokens: int = 100,
    output_tokens: int = 10,
    cache_read_tokens: int = 0,
    details: dict[str, int] | None = None,
) -> RequestUsage:
    return RequestUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        details=details or {},
    )


def tool_result(
    info: AgentInfo,
    name: str,
    payload: dict[str, Any],
    *,
    token_usage: RequestUsage | None = None,
) -> ModelResponse:
    match = next(tool for tool in info.output_tools if tool.name == name)
    return ModelResponse(
        parts=[ToolCallPart(match.name, payload)],
        usage=token_usage or usage(cache_read_tokens=80),
    )


def text_result(payload: dict[str, Any], *, token_usage: RequestUsage | None = None) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(json.dumps(payload))],
        usage=token_usage or usage(cache_read_tokens=80),
    )


def user_text(messages: list[ModelMessage]) -> str:
    chunks: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                chunks.append(content)
    return "\n".join(chunks)
