"""Failure classification.

Every model call ends as either a value or a :class:`Failure`. Nothing is dropped
on the floor: transport and validation failures are retryable up to the policy
bound, and permanent failures are returned to the caller.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import ValidationError
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)

try:
    import httpx
except ImportError:  # pragma: no cover - pydantic-ai always brings an http client
    httpx = None  # type: ignore[assignment]


class FailureKind(StrEnum):
    """Coarse class used for retry decisions and Temporal non-retryable mapping."""

    VALIDATION = "validation"
    TRANSPORT = "transport"
    PERMANENT = "permanent"
    BUDGET = "budget"


# 408/429 and the usual gateway/unavailable statuses are worth another attempt.
# Other 4xx responses (auth, bad request, not found) will not succeed if repeated.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass
class Failure:
    """A classified failure attached to a query or a run.

    ``retryable`` is the executor's signal. Temporal maps it to
    ``ApplicationError(non_retryable=not retryable)``.
    """

    kind: FailureKind
    message: str
    retryable: bool
    code: str = ""
    attempts: int = 1
    detail: dict[str, Any] = field(default_factory=dict)

    def with_attempts(self, attempts: int) -> Failure:
        return Failure(
            kind=self.kind,
            message=self.message,
            retryable=self.retryable,
            code=self.code,
            attempts=attempts,
            detail=dict(self.detail),
        )


def classify_exception(exc: BaseException) -> Failure:
    """Map an exception from a model call onto a :class:`Failure`."""
    message = _short_message(exc)
    if isinstance(exc, ModelHTTPError):
        overflow = _context_overflow_from_provider(exc)
        if overflow is not None:
            return overflow
        retryable = exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
        kind = FailureKind.TRANSPORT if retryable else FailureKind.PERMANENT
        return Failure(
            kind=kind,
            message=message,
            retryable=retryable,
            code=f"http_{exc.status_code}",
            detail={"status_code": exc.status_code},
        )
    if isinstance(exc, ModelAPIError):
        return Failure(
            kind=FailureKind.TRANSPORT,
            message=message,
            retryable=True,
            code="transport",
        )
    if httpx is not None and isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return Failure(
            kind=FailureKind.TRANSPORT,
            message=message,
            retryable=True,
            code="transport",
        )
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return Failure(
            kind=FailureKind.TRANSPORT,
            message=message,
            retryable=True,
            code="transport",
        )
    if isinstance(exc, (UnexpectedModelBehavior, ValidationError, UsageLimitExceeded)):
        return Failure(
            kind=FailureKind.VALIDATION,
            message=message,
            retryable=True,
            code="validation",
        )
    if isinstance(exc, UserError):
        return Failure(
            kind=FailureKind.PERMANENT,
            message=message,
            retryable=False,
            code="user_error",
        )
    return Failure(
        kind=FailureKind.PERMANENT,
        message=message,
        retryable=False,
        code="unexpected",
        detail={"exception_type": type(exc).__name__},
    )


def wrong_output_failure(expected: str, actual: str) -> Failure:
    """The model called a registered output tool other than the one this query asked for."""
    return Failure(
        kind=FailureKind.VALIDATION,
        message=f"Expected output {expected}, got {actual}",
        retryable=True,
        code="wrong_output_type",
        detail={"expected": expected, "actual": actual},
    )


def context_overflow_failure(
    *,
    estimated_tokens: int | None,
    limit: int | None,
    where: str,
    reported_tokens: int | None = None,
) -> Failure:
    """Permanent failure when one prompt does not fit the context window.

    The library does not truncate or split the prompt. ``where`` is ``prefix``
    at build time and ``request`` for a single query.
    """
    size = reported_tokens if reported_tokens is not None else estimated_tokens
    size_label = "reported" if reported_tokens is not None else "estimated"
    size_text = "unknown" if size is None else str(size)
    limit_text = "unknown" if limit is None else str(limit)
    detail: dict[str, Any] = {"where": where}
    if estimated_tokens is not None:
        detail["estimated_tokens"] = estimated_tokens
    if reported_tokens is not None:
        detail["reported_tokens"] = reported_tokens
    if limit is not None:
        detail["limit"] = limit
    return Failure(
        kind=FailureKind.PERMANENT,
        message=(f"context overflow: {size_label} {size_text} tokens exceeds context limit {limit_text} ({where})"),
        retryable=False,
        code="context_overflow",
        detail=detail,
    )


def budget_failure(message: str) -> Failure:
    return Failure(
        kind=FailureKind.BUDGET,
        message=message,
        retryable=False,
        code="budget_exceeded",
    )


_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "context_overflow",
    "maximum context length",
    "context window",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "exceeds the context",
    "exceeds the model's maximum context",
)


def _context_overflow_from_provider(exc: ModelHTTPError) -> Failure | None:
    text = _provider_error_text(exc)
    if not any(marker in text.lower() for marker in _OVERFLOW_MARKERS):
        return None
    reported, reported_limit = _reported_context_sizes(text)
    failure = context_overflow_failure(
        estimated_tokens=None,
        limit=reported_limit,
        where="provider",
        reported_tokens=reported,
    )
    failure.detail["status_code"] = exc.status_code
    return failure


def _provider_error_text(exc: ModelHTTPError) -> str:
    body = exc.body
    if isinstance(body, str):
        rendered = body
    elif body is None:
        rendered = ""
    else:
        rendered = str(body)
    return f"{exc} {rendered}"


def _reported_context_sizes(text: str) -> tuple[int | None, int | None]:
    """Pull ``(requested tokens, window)`` out of a provider error, when present."""
    match = re.search(
        r"maximum context length is (\d+).*?requested (\d+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None, None
    return int(match.group(2)), int(match.group(1))


def _short_message(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    text = " ".join(text.split())
    if len(text) > 500:
        text = text[:500] + "..."
    return f"{type(exc).__name__}: {text}"
