"""Frozen prefix identity.

Providers render tool definitions before the message list, so the cacheable head
of a request is:

    tool set + instructions + cleaned documents + end marker + query tail

Tool definitions are not message-history entries. ``prefix_hash`` still covers
them, together with the instructions and the rendered documents, so two sessions
with different schemas do not share an idempotency key. The message-history
prefix that continuations must start with is the rendered document request.

Raw documents omit wrappers and the end marker. Single layout freezes the
document text plus ``QUERY_SEPARATOR`` before each branch's initial query tail;
continuations retain that combined original user part.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Sequence
from typing import Any, Literal

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    UserPromptPart,
)

from forkreduce.documents import END_MARKER, DocumentFormat
from forkreduce.errors import ForkReduceError, PrefixMismatchWarning

MessageLayout = Literal["split", "single"]
QUERY_SEPARATOR = "\n\n===QUERY===\n\n"


def prefix_hash(
    *,
    output_mode: str,
    instructions: str,
    rendered_documents: str,
    tools: Sequence[dict[str, Any]],
    document_format: DocumentFormat = "blocks",
    message_layout: MessageLayout = "split",
    query_separator: str = QUERY_SEPARATOR,
) -> str:
    """Stable SHA-256 over the frozen prefix contract."""
    if document_format not in ("blocks", "raw"):
        raise ForkReduceError("document_format must be 'blocks' or 'raw'")
    _check_message_layout(message_layout)
    _check_query_separator(query_separator)
    payload = {
        "version": 1,
        "output_mode": output_mode,
        "instructions": instructions,
        "documents": rendered_documents,
        "tools": list(tools),
    }
    if document_format != "blocks":
        payload["document_format"] = document_format
    if message_layout != "split":
        payload["message_layout"] = message_layout
    if query_separator != QUERY_SEPARATOR:
        payload["query_separator"] = query_separator
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prefix_messages(
    rendered_documents: str,
    *,
    message_layout: MessageLayout = "split",
    tail: str | None = None,
    query_separator: str = QUERY_SEPARATOR,
) -> list[ModelMessage]:
    """Fresh document history, including the initial tail in single layout."""
    _check_message_layout(message_layout)
    _check_query_separator(query_separator)
    if message_layout == "single":
        if tail is None:
            raise ForkReduceError("single message_layout requires a query tail")
        rendered_documents += query_separator + tail
    return [ModelRequest(parts=[UserPromptPart(content=rendered_documents)])]


def history_matches_prefix(
    history: Sequence[ModelMessage],
    rendered_documents: str,
    *,
    message_layout: MessageLayout = "split",
    query_separator: str = QUERY_SEPARATOR,
) -> bool:
    """Whether ``history`` starts with the frozen document text.

    Timestamps, run ids, and instructions stamped onto the request by pydantic-ai
    are ignored. Only the first user-prompt text is compared.
    """
    _check_message_layout(message_layout)
    _check_query_separator(query_separator)
    if not history:
        return False
    first = history[0]
    if not isinstance(first, ModelRequest):
        return False
    for part in first.parts:
        if isinstance(part, UserPromptPart) and isinstance(part.content, str):
            text = part.content
            if message_layout == "single":
                return text.startswith(rendered_documents + query_separator)
            return text == rendered_documents or text.startswith(rendered_documents)
    return False


def warn_if_prefix_mismatch(
    history: Sequence[ModelMessage] | None,
    rendered_documents: str,
    *,
    prefix_hash_value: str,
    message_layout: MessageLayout = "split",
    query_separator: str = QUERY_SEPARATOR,
) -> None:
    if history is None:
        return
    if history_matches_prefix(
        history, rendered_documents, message_layout=message_layout, query_separator=query_separator
    ):
        return
    marker = query_separator.strip() if message_layout == "single" else END_MARKER
    warnings.warn(
        "message history does not start with the frozen document prefix "
        f"(prefix_hash={prefix_hash_value}, marker={marker}). "
        "The call will run, but the provider prefix cache will miss.",
        PrefixMismatchWarning,
        stacklevel=3,
    )


def _check_message_layout(message_layout: MessageLayout) -> None:
    if message_layout not in ("split", "single"):
        raise ForkReduceError("message_layout must be 'split' or 'single'")


def _check_query_separator(query_separator: str) -> None:
    if not isinstance(query_separator, str):
        raise ForkReduceError("query_separator must be a string")
