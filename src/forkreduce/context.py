"""Per-call context window check.

The estimate is ``ceil(characters / 4)`` over the instructions, the tool schemas,
the documents or continuation history, and the query tail. It is not a provider
tokenizer. A prompt over :attr:`forkreduce.policy.Policy.context_limit` fails
before the model is called. A provider that still reports a context-length error
fails the same way. Nothing is truncated or split into chunks.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import ModelMessage

from forkreduce.failures import Failure, context_overflow_failure
from forkreduce.policy import Policy


def estimate_tokens(text: str) -> int:
    """Rough token estimate: one token per four characters, rounded up."""
    if not text:
        return 0
    return (len(text) + 3) // 4


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


def check_context(*, estimated_tokens: int, limit: int | None, where: str) -> Failure | None:
    """Return a non-retryable failure when ``estimated_tokens`` is over ``limit``."""
    if limit is None or estimated_tokens <= limit:
        return None
    return context_overflow_failure(estimated_tokens=estimated_tokens, limit=limit, where=where)


def check_policy_context(policy: Policy, *, estimated_tokens: int, where: str) -> Failure | None:
    return check_context(estimated_tokens=estimated_tokens, limit=policy.context_limit, where=where)
