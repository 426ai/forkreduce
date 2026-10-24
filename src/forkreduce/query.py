"""Queries and the result of running one branch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from forkreduce.context import ContextWarning
from forkreduce.failures import Failure
from forkreduce.output_modes import ResolvedOutputMode
from forkreduce.output_specs import OutputSpec

T = TypeVar("T")


@dataclass(frozen=True)
class OutputValidationContext:
    """Worker-local query identity passed to contextual output validators.

    ``meta`` is a deep copy of the caller's query metadata. This context is
    serializable independently of the worker-local validator callback.
    """

    query_id: str
    round: int
    meta: dict[str, Any]


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

    ``output`` is a registered BaseModel subclass or a per-query ``OutputSpec``.
    Dynamic outputs produce dictionaries; tools mode requires a generic output
    tool registered when building the session.

    ``model_settings`` overrides the session settings for this call only.
    pydantic-ai merges the two shallowly: a key set here replaces that key,
    and ``extra_body`` is replaced as a whole rather than merged key by key.
    In an auto session, overrides that would change the resolved output mode
    raise ``OutputModeConflictError`` before budget reservation or model calls.
    """

    tail: str
    output: type[T] | OutputSpec
    id: str | None = None
    history: list[ModelMessage] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    round: int = 0
    model_settings: ModelSettings | None = None


@dataclass
class QueryResult(Generic[T]):
    """The outcome of one query, including the branch transcript.

    ``messages`` is the pydantic-ai ``all_messages()`` list on success, so a
    caller can continue the branch. On a failure that raised before a transcript
    existed, ``messages`` is the history that was sent.

    ``needs_review`` and ``dropped`` are set by check/repair policy. A dropped
    result stays on the run; it is only left out of reduce inputs.

    ``context_warning`` is set when the local token estimate exceeds
    ``Policy.context_limit``. The call was still sent. A provider rejection is
    ``failure`` with code ``context_overflow``, not this field.

    ``resolved_output_mode`` records the actual session mode on successes and
    failures. ``None`` is retained only for legacy/manually constructed results.

    ``feedback_retries_used`` counts additional model requests sent for native
    output validation, summed across local executor attempts. ``attempts`` keeps
    its outer-attempt meaning. ``meta`` always contains the same feedback count;
    session-created metadata does not copy query text or caller query metadata.
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
    context_warning: ContextWarning | None = None
    resolved_output_mode: ResolvedOutputMode | None = None
    feedback_retries_used: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.meta = {**self.meta, "feedback_retries_used": self.feedback_retries_used}
