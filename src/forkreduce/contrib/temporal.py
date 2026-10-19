"""Temporal adapter.

Each query is one activity. The idempotency key is ``prefix_hash + query_id``,
exposed as :func:`activity_id`. The activity performs a single model attempt and
raises :class:`temporalio.exceptions.ApplicationError` with ``non_retryable``
set from the failure class, so Temporal's retry policy is the bound.

Build the session inside the worker and :func:`register_session` it. The
workflow payload carries the prefix hash and the query tail, not the model.

Large transcripts should go through a :class:`MessageStore`. Without one, the
activity inlines the messages and sets ``messages_json``; that is convenient
for tests and a bad idea for real histories.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel
    from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

    from forkreduce.errors import ForkReduceError
    from forkreduce.failures import Failure
    from forkreduce.policy import Policy
    from forkreduce.query import Query
    from forkreduce.session import Session


class MessageStore(Protocol):
    """Store branch transcripts outside the activity result."""

    async def put(self, key: str, payload: bytes) -> str:
        """Store ``payload`` and return a reference the worker can resolve."""
        ...

    async def get(self, ref: str) -> bytes:
        """Load the payload stored under ``ref``."""
        ...


class InMemoryMessageStore:
    """Process-local store. One worker only; not a shared production store."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._seq = 0

    async def put(self, key: str, payload: bytes) -> str:
        self._seq += 1
        ref = f"{key}:{self._seq}"
        self._data[ref] = payload
        return ref

    async def get(self, ref: str) -> bytes:
        try:
            return self._data[ref]
        except KeyError as exc:
            raise KeyError(f"unknown message ref {ref!r}") from exc


_SESSIONS: dict[str, Session] = {}
_FINGERPRINTS: dict[str, str] = {}
_STORE: MessageStore | None = None


def execution_fingerprint(session: Session) -> str:
    """Identity of how a session runs, beyond the frozen prompt prefix.

    ``prefix_hash`` ignores the model and the retry/budget policy. Two sessions
    can share a hash and still send calls to different places.
    """
    model = session.model
    policy = session.policy
    budget = policy.budget
    payload = {
        "model_type": type(model).__qualname__,
        "model_name": getattr(model, "model_name", None),
        "system": getattr(model, "system", None),
        "output_mode": session.output_mode,
        "warmup": session.warmup,
        "concurrency": policy.concurrency,
        "retries": policy.retries,
        "retry_backoff": policy.retry_backoff,
        "max_backoff": policy.max_backoff,
        "repair_failure": policy.repair_failure,
        "context_limit": policy.context_limit,
        "context_reserve": policy.context_reserve,
        "budget": None
        if budget is None
        else {
            "max_requests": budget.max_requests,
            "max_input_tokens": budget.max_input_tokens,
            "max_output_tokens": budget.max_output_tokens,
            "max_total_tokens": budget.max_total_tokens,
        },
    }
    return json.dumps(payload, sort_keys=True, default=str)


def register_session(session: Session) -> str:
    """Make ``session`` available to activities in this process. Returns its prefix hash.

    A second registration with the same prefix and the same execution fingerprint
    replaces the session. A different model or policy for that prefix raises
    :class:`forkreduce.errors.ForkReduceError` instead of silently taking over.
    """
    key = session.prefix_hash
    fingerprint = execution_fingerprint(session)
    current = _FINGERPRINTS.get(key)
    if current is not None and current != fingerprint:
        raise ForkReduceError(
            "A session with this prefix_hash is already registered with a different model or policy. "
            "Call unregister_session before registering a different execution config."
        )
    _SESSIONS[key] = session
    _FINGERPRINTS[key] = fingerprint
    return key


def unregister_session(prefix_hash: str) -> None:
    """Drop a process-local registration. Unknown hashes are ignored."""
    _SESSIONS.pop(prefix_hash, None)
    _FINGERPRINTS.pop(prefix_hash, None)


def set_message_store(store: MessageStore | None) -> None:
    """Install the store activities use for transcripts. ``None`` inlines messages."""
    global _STORE
    _STORE = store


def activity_id(prefix_hash: str, query_id: str) -> str:
    """Idempotency key for one query on one frozen prefix."""
    return f"forkreduce:{prefix_hash}:{query_id}"


def to_application_error(failure: Failure) -> ApplicationError:
    """Map a classified failure onto a Temporal application error.

    Validation and transport failures stay retryable. Permanent failures and
    budget exhaustion are ``non_retryable``.
    """
    return ApplicationError(
        failure.message,
        failure.kind.value,
        type=failure.kind.value,
        non_retryable=not failure.retryable,
    )


def temporal_retry_policy(policy: Policy) -> RetryPolicy:
    """A Temporal retry policy with the same attempt budget as ``Policy.retries``."""
    initial = policy.retry_backoff if policy.retry_backoff > 0 else 0.001
    maximum = policy.max_backoff if policy.max_backoff > 0 else initial
    return RetryPolicy(
        initial_interval=timedelta(seconds=initial),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(seconds=maximum),
        maximum_attempts=policy.retries + 1,
    )


@dataclass
class QueryActivityInput:
    """Serializable query. The worker looks up the session by ``prefix_hash``.

    ``max_attempts`` is ``Policy.retries + 1``. :class:`MapQueriesWorkflow` reads
    it instead of assuming three attempts. ``history_json`` continues a branch
    when no :class:`MessageStore` is configured; it is the previous activity's
    ``messages_json``.
    """

    prefix_hash: str
    query_id: str
    tail: str
    output_name: str
    round: int = 0
    history_ref: str | None = None
    history_json: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 1
    retry_backoff_seconds: float = 0.0
    max_backoff_seconds: float = 0.0


def query_activity_input(
    session: Session,
    *,
    query_id: str,
    tail: str,
    output_name: str,
    round: int = 0,
    history_ref: str | None = None,
    history_json: str | None = None,
    meta: dict[str, Any] | None = None,
) -> QueryActivityInput:
    """Build an activity payload that carries this session's retry policy."""
    policy = session.policy
    return QueryActivityInput(
        prefix_hash=session.prefix_hash,
        query_id=query_id,
        tail=tail,
        output_name=output_name,
        round=round,
        history_ref=history_ref,
        history_json=history_json,
        meta=dict(meta or {}),
        max_attempts=policy.retries + 1,
        retry_backoff_seconds=policy.retry_backoff,
        max_backoff_seconds=policy.max_backoff,
    )


def activity_retry_policy(payload: QueryActivityInput) -> RetryPolicy:
    """Retry policy for one activity, taken from its payload."""
    initial = payload.retry_backoff_seconds if payload.retry_backoff_seconds > 0 else 0.001
    maximum = payload.max_backoff_seconds if payload.max_backoff_seconds > 0 else initial
    return RetryPolicy(
        initial_interval=timedelta(seconds=initial),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(seconds=maximum),
        maximum_attempts=max(payload.max_attempts, 1),
    )


@dataclass
class QueryActivityOutput:
    """Activity result. Messages are a ref when a store is configured."""

    query_id: str
    prefix_hash: str
    output_json: str | None
    output_type: str | None
    cache_hit_tokens: int
    attempts: int
    messages_ref: str | None = None
    messages_json: str | None = None


async def execute_query_activity(payload: QueryActivityInput) -> QueryActivityOutput:
    """Run one query attempt. Temporal activities call this through the defn wrapper."""
    session = _SESSIONS.get(payload.prefix_hash)
    if session is None:
        raise ApplicationError(
            f"No session registered for prefix {payload.prefix_hash}",
            "permanent",
            type="permanent",
            non_retryable=True,
        )
    output_type = session.output_types.get(payload.output_name)
    if output_type is None:
        raise ApplicationError(
            f"Output {payload.output_name!r} is not registered on this session",
            "permanent",
            type="permanent",
            non_retryable=True,
        )
    history: list[ModelMessage] | None = None
    if payload.history_ref is not None and payload.history_json is not None:
        raise ApplicationError(
            "Set only one of history_ref and history_json",
            "permanent",
            type="permanent",
            non_retryable=True,
        )
    if payload.history_json is not None:
        history = list(ModelMessagesTypeAdapter.validate_json(payload.history_json))
    elif payload.history_ref is not None:
        if _STORE is None:
            raise ApplicationError(
                "history_ref was set but this worker has no message store",
                "permanent",
                type="permanent",
                non_retryable=True,
            )
        history = list(ModelMessagesTypeAdapter.validate_json(await _STORE.get(payload.history_ref)))
    query: Query[Any] = Query(
        tail=payload.tail,
        output=output_type,
        id=payload.query_id,
        history=history,
        meta=dict(payload.meta),
        round=payload.round,
    )
    session.prepare_query(query)
    result = await session.run_once(query)
    if result.failure is not None:
        raise to_application_error(result.failure)
    dumped = ModelMessagesTypeAdapter.dump_json(result.messages)
    messages_ref: str | None = None
    messages_json: str | None = None
    if _STORE is not None:
        messages_ref = await _STORE.put(f"{payload.prefix_hash}:{payload.query_id}", dumped)
    else:
        messages_json = dumped.decode()
    output_json: str | None = None
    output_name: str | None = None
    if isinstance(result.output, BaseModel):
        output_json = result.output.model_dump_json()
        output_name = type(result.output).__name__
    return QueryActivityOutput(
        query_id=result.query_id,
        prefix_hash=result.prefix_hash,
        output_json=output_json,
        output_type=output_name,
        cache_hit_tokens=result.cache_hit_tokens,
        attempts=result.attempts,
        messages_ref=messages_ref,
        messages_json=messages_json,
    )


@activity.defn(name="forkreduce.run_query")
async def run_query_activity(payload: QueryActivityInput) -> QueryActivityOutput:
    """Temporal activity entrypoint. One attempt; raise to retry."""
    return await execute_query_activity(payload)


@workflow.defn(name="forkreduce.map_queries")
class MapQueriesWorkflow:
    """Fan out query activities. Deterministic: it only schedules activities.

    Coverage math and other pure decisions can run in the workflow after the
    activities return. Model calls, including reduce and gap-fill, stay in
    activities. Register the session on the worker before starting this workflow.
    """

    @workflow.run
    async def run(self, inputs: list[QueryActivityInput]) -> list[QueryActivityOutput]:
        async def one(payload: QueryActivityInput) -> QueryActivityOutput:
            return await workflow.execute_activity(
                run_query_activity,
                payload,
                start_to_close_timeout=timedelta(minutes=5),
                activity_id=activity_id(payload.prefix_hash, payload.query_id),
                retry_policy=activity_retry_policy(payload),
            )

        return list(await asyncio.gather(*(one(payload) for payload in inputs)))
