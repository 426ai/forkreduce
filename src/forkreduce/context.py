"""Local context estimate.

The estimate is ``ceil(characters / 4)`` over the instructions, the tool schemas,
the documents or continuation history, and the query tail. It is not a provider
tokenizer, and it never blocks a call or truncates a prompt. When it exceeds the
configured window, forkreduce warns and still sends the request. Only a
provider-reported context-length error becomes a ``context_overflow`` failure.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ModelMessage

from forkreduce.policy import Policy

logger = logging.getLogger(__name__)


class ContextEstimateWarning(UserWarning):
    """The local token estimate is over the configured context window.

    The request is still sent. The provider decides whether it actually fits.
    """


@dataclass(frozen=True)
class ContextWarning:
    """A local estimate that exceeded the configured window.

    ``where`` is ``prefix`` at build time and ``request`` for one query.
    ``reserve`` is the headroom held back from the prefix; request warnings
    compare against the full ``limit`` and still record ``reserve``.
    """

    where: str
    estimated_tokens: int
    limit: int
    reserve: int
    message: str


def estimate_tokens(text: str) -> int:
    """Rough token estimate: one token per four characters, rounded up."""
    if not text:
        return 0
    return (len(text) + 3) // 4


def effective_context_reserve(policy: Policy) -> int:
    """Tokens of headroom used when deciding whether a prefix should warn.

    ``Policy.context_reserve is None`` selects
    ``min(context_limit, max(4096, context_limit // 4))`` when a limit is set,
    and 0 when it is not. An explicit integer, including 0, is used as given.
    """
    if policy.context_reserve is not None:
        return policy.context_reserve
    limit = policy.context_limit
    if limit is None:
        return 0
    return min(limit, max(4096, limit // 4))


def prefix_text(instructions: str, rendered_documents: str, tool_specs: Sequence[dict[str, Any]]) -> str:
    """Text standing in for the frozen prefix, including tool schemas."""
    tools = json.dumps(list(tool_specs), sort_keys=True, default=str)
    return f"{instructions}\n{rendered_documents}\n{tools}"


def history_text(messages: Sequence[ModelMessage]) -> str:
    """Flatten a branch transcript into the characters that count toward the window."""
    chunks: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                chunks.append(content)
            elif content is not None:
                chunks.append(json.dumps(content, sort_keys=True, default=str))
    return "\n".join(chunks)


def request_text(
    *,
    instructions: str,
    rendered_documents: str,
    tool_specs: Sequence[dict[str, Any]],
    tail: str,
    history: Sequence[ModelMessage] | None,
) -> str:
    """Characters for one model call.

    Instructions and tool schemas are sent on every call even when they are not
    in the message history. Continuation history already contains the documents,
    so those are not added again.
    """
    tools = json.dumps(list(tool_specs), sort_keys=True, default=str)
    body = history_text(history) if history else rendered_documents
    return f"{instructions}\n{tools}\n{body}\n{tail}"


def prefix_context_warning(policy: Policy, estimated_tokens: int) -> ContextWarning | None:
    """Warn when the frozen prefix exceeds ``context_limit - reserve``.

    ``context_limit is None`` skips the local warning. Equality is inside the window.
    """
    limit = policy.context_limit
    if limit is None:
        return None
    reserve = effective_context_reserve(policy)
    usable = limit - reserve
    if estimated_tokens <= usable:
        return None
    return ContextWarning(
        where="prefix",
        estimated_tokens=estimated_tokens,
        limit=limit,
        reserve=reserve,
        message=(
            f"context estimate: estimated {estimated_tokens} prefix tokens exceeds usable limit "
            f"{usable} (context limit {limit} minus reserve {reserve})"
        ),
    )


def request_context_warning(policy: Policy, estimated_tokens: int) -> ContextWarning | None:
    """Warn when one full request estimate exceeds ``context_limit``.

    The reserve is not subtracted here: the tail is already part of the estimate.
    It is still recorded on the warning. ``context_limit is None`` skips the warning.
    """
    limit = policy.context_limit
    if limit is None or estimated_tokens <= limit:
        return None
    reserve = effective_context_reserve(policy)
    return ContextWarning(
        where="request",
        estimated_tokens=estimated_tokens,
        limit=limit,
        reserve=reserve,
        message=(
            f"context estimate: estimated {estimated_tokens} tokens exceeds context limit {limit} (reserve {reserve})"
        ),
    )


def emit_context_warning(warning: ContextWarning, *, stacklevel: int = 2) -> None:
    """Emit ``warning`` through ``warnings`` and the ``forkreduce.context`` logger."""
    warnings.warn(warning.message, ContextEstimateWarning, stacklevel=stacklevel)
    logger.warning(warning.message)
