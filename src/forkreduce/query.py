"""Queries and the result of running one branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from forkreduce.failures import Failure

T = TypeVar("T")


@dataclass(frozen=True)
class CheckFailure:
    """One problem found by a :class:`forkreduce.checks.Check`."""

    code: str
    message: str
    path: str
    block_id: str | None = None
    quote: str | None = None


@dataclass
class Query(Generic[T]):
    """One branch off the frozen prefix.

    ``history is None`` starts a new branch at the shared prefix. Pass
    ``history=previous.messages`` to continue that branch (citation repair and
    single-question follow-ups). The framework does not decide which one you
    wanted; it only warns when a continuation does not start with the prefix.

    ``output`` must be one of the types registered on the session.
    """

    tail: str
    output: type[T]
    id: str | None = None
    history: list[ModelMessage] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    round: int = 0


@dataclass
class QueryResult(Generic[T]):
    """The outcome of one query, including the branch transcript.

    ``messages`` is the pydantic-ai ``all_messages()`` list on success, so a
    caller can continue the branch. On a failure that raised before a transcript
    existed, ``messages`` is the history that was sent.

    ``needs_review`` and ``dropped`` are set by check/repair policy. A dropped
    result stays on the run; it is only left out of reduce inputs.
    """

    query_id: str
    output: T | None
    messages: list[ModelMessage]
    prefix_hash: str
    usage: RunUsage
    cache_hit_tokens: int
    attempts: int
    failure: Failure | None
    round: int
    elapsed_ms: float = 0.0
    needs_review: bool = False
    dropped: bool = False
    check_failures: list[CheckFailure] = field(default_factory=list)
