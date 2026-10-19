"""Frozen prefix identity.

Providers render tool definitions before the message list, so the cacheable head
of a request is:

    tool set + instructions + cleaned documents + end marker + query tail

Tool definitions are not message-history entries. ``prefix_hash`` still covers
them, together with the instructions and the rendered documents, so two sessions
with different schemas do not share an idempotency key. The message-history
prefix that continuations must start with is the rendered document request.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    UserPromptPart,
)

from forkreduce.documents import END_MARKER
from forkreduce.errors import PrefixMismatchWarning


def prefix_hash(
    *,
    output_mode: str,
    instructions: str,
    rendered_documents: str,
    tools: Sequence[dict[str, Any]],
) -> str:
    """Stable SHA-256 over the frozen prefix contract."""
    payload = {
        "version": 1,
        "output_mode": output_mode,
        "instructions": instructions,
        "documents": rendered_documents,
        "tools": list(tools),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prefix_messages(rendered_documents: str) -> list[ModelMessage]:
    """A fresh copy of the document prefix. Do not reuse the list across calls."""
    return [ModelRequest(parts=[UserPromptPart(content=rendered_documents)])]


def history_matches_prefix(history: Sequence[ModelMessage], rendered_documents: str) -> bool:
    """Whether ``history`` starts with the frozen document text.

    Timestamps, run ids, and instructions stamped onto the request by pydantic-ai
    are ignored. Only the first user-prompt text is compared.
    """
    if not history:
        return False
    first = history[0]
    if not isinstance(first, ModelRequest):
        return False
    for part in first.parts:
        if isinstance(part, UserPromptPart) and isinstance(part.content, str):
            text = part.content
            return text == rendered_documents or text.startswith(rendered_documents)
    return False


def warn_if_prefix_mismatch(
    history: Sequence[ModelMessage] | None,
    rendered_documents: str,
    *,
    prefix_hash_value: str,
) -> None:
    if history is None:
        return
    if history_matches_prefix(history, rendered_documents):
        return
    warnings.warn(
        "message history does not start with the frozen document prefix "
        f"(prefix_hash={prefix_hash_value}, marker={END_MARKER}). "
        "The call will run, but the provider prefix cache will miss.",
        PrefixMismatchWarning,
        stacklevel=3,
    )
