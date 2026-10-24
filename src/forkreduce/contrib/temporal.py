"""Temporal adapter.

Each query is one activity. The idempotency key is ``prefix_hash + query_id``,
exposed as :func:`activity_id`. The activity performs a single model attempt and
raises :class:`temporalio.exceptions.ApplicationError` with ``non_retryable``
set from the failure class, so Temporal's retry policy is the bound.

Register a worker-owned session with :func:`register_session`, or persist its
prefix using :func:`freeze_prefix` and register a :func:`register_session_factory`.
Factories reconstruct the model and output classes locally; neither is serialized.

Transcripts use :class:`MessageStore` references when configured. Direct maps
retain inline results for compatibility; child batches require references for
both transcripts and structured outputs. Direct callers can request the same
reference-only behavior with ``QueryActivityInput.require_references``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from typing import Any, Protocol, cast

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic import BaseModel
    from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
    from pydantic_ai.settings import ModelSettings, merge_model_settings

    from forkreduce.documents import Document
    from forkreduce.errors import ForkReduceError
    from forkreduce.failures import Failure
    from forkreduce.models import canonical_value, request_body_settings
    from forkreduce.output_modes import OutputMode, ResolvedOutputMode, UnforcedToolsPolicy
    from forkreduce.policy import Policy
    from forkreduce.prefix import prefix_hash as compute_prefix_hash
    from forkreduce.query import Query
    from forkreduce.session import Session
    from forkreduce.temporal_options import ActivityOptions


class MessageStore(Protocol):
    """Store branch transcripts outside the activity result."""

    async def put(self, key: str, payload: bytes) -> str:
        """Store ``payload`` and return a reference the worker can resolve."""
        ...

    async def get(self, ref: str) -> bytes:
        """Load the payload stored under ``ref``."""
        ...


class PrefixStore(MessageStore, Protocol):
    """Persist frozen prefixes for reconstruction on any activity worker."""


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


@dataclass(frozen=True)
class FrozenPrefix:
    """JSON-only snapshot; worker factories supply models and output classes.

    ``documents_json`` preserves cleaned document records. ``rendered_documents``
    preserves the exact prefix, including any custom metadata rendering.
    A factory must reproduce custom metadata rendering as well as the model and
    output schemas: the reconstructed session's complete prefix is verified.
    """

    prefix_hash: str
    instructions: str
    documents_json: str
    rendered_documents: str
    output_mode: OutputMode
    tools_json: str
    resolved_output_mode: ResolvedOutputMode | None = None
    on_unforced_tools: UnforcedToolsPolicy = "warn"

    @property
    def documents(self) -> list[Document]:
        """Fresh document objects suitable for ``build(documents=...)``."""
        return [Document.model_validate(item) for item in json.loads(self.documents_json)]

    def validate(self) -> None:
        """Reject corrupt snapshots before invoking a worker factory."""
        resolved = self.resolved_output_mode
        if resolved is None:
            if self.output_mode not in {"tools", "prompted"}:
                raise ValueError("An auto snapshot must record its resolved_output_mode")
            resolved = self.output_mode
        if resolved not in {"tools", "prompted"}:
            raise ValueError("Invalid resolved_output_mode in frozen prefix")
        if self.output_mode not in {"auto", resolved}:
            raise ValueError("Frozen requested and resolved output modes conflict")
        actual = compute_prefix_hash(
            output_mode=resolved,
            instructions=self.instructions,
            rendered_documents=self.rendered_documents,
            tools=json.loads(self.tools_json),
        )
        if actual != self.prefix_hash:
            raise ValueError("Frozen prefix hash does not match its contents")
        for item in json.loads(self.documents_json):
            Document.model_validate(item)


SessionFactory = Callable[[FrozenPrefix], Session | Awaitable[Session]]
_FACTORIES: dict[str, SessionFactory] = {}
_PREFIX_STORE: PrefixStore | None = None


async def freeze_prefix(session: Session, store: PrefixStore) -> str:
    """Persist a prefix without serializing the model or Python output classes."""
    snapshot = FrozenPrefix(
        prefix_hash=session.prefix_hash,
        instructions=session.instructions,
        documents_json=json.dumps([doc.model_dump(mode="json") for doc in session.documents.documents]),
        rendered_documents=session.rendered_documents,
        output_mode=session.output_mode,
        tools_json=json.dumps(session.tool_specs),
        resolved_output_mode=session.resolved_output_mode,
        on_unforced_tools=session.on_unforced_tools,
    )
    snapshot.validate()
    return await store.put(f"prefix:{session.prefix_hash}", json.dumps(asdict(snapshot)).encode())


def set_prefix_store(store: PrefixStore | None) -> None:
    """Install the worker store used to resolve ``QueryActivityInput.prefix_ref``."""
    global _PREFIX_STORE
    _PREFIX_STORE = store


def register_session_factory(prefix_hash: str, factory: SessionFactory) -> None:
    """Register a sync or async factory; missing configuration fails permanently.

    Unless a session is explicitly registered, the factory runs on every activity
    attempt. The adapter does not implicitly share session budgets across those
    attempts or unrelated workflows. A factory can cache its Session or call
    :func:`register_session` when worker-wide reuse and budget sharing are wanted.
    """
    _FACTORIES[prefix_hash] = factory


def execution_fingerprint(session: Session) -> str:
    """Identity of how a session runs, beyond the frozen prompt prefix.

    ``prefix_hash`` ignores the model and the retry/budget policy. Two sessions
    can share a hash and still send calls to different places.

    The fingerprint includes settings that change the request body (sampling
    parameters, ``extra_body``, and anything else on ``ModelSettings`` except
    ``timeout`` and ``extra_headers``) and the resolved model profile. A
    profile can change tool or schema rendering, so two sessions with the same
    prefix hash and different profiles are different executions. A per-query
    ``model_settings`` override is not part of this identity; the activity
    payload carries that override itself.

    Sampling parameters, timeouts, and extra headers do not change the frozen
    prefix bytes. A profile, or opt-in block meta, can.
    """
    model = session.model
    policy = session.policy
    budget = policy.budget
    raw_settings = getattr(model, "settings", None)
    model_settings = cast(ModelSettings, raw_settings) if isinstance(raw_settings, Mapping) else None
    merged_settings = merge_model_settings(model_settings, session.model_settings)
    payload = {
        "model_type": type(model).__qualname__,
        "model_name": getattr(model, "model_name", None),
        "system": getattr(model, "system", None),
        "model_settings": canonical_value(request_body_settings(merged_settings)),
        "model_profile": canonical_value(getattr(model, "profile", None)),
        "output_mode": session.output_mode,
        "resolved_output_mode": session.resolved_output_mode,
        "on_unforced_tools": session.on_unforced_tools,
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
            "A session with this prefix_hash is already registered with a different model, "
            "model settings, profile, or policy. "
            "Call unregister_session before registering a different execution config."
        )
    _SESSIONS[key] = session
    _FINGERPRINTS[key] = fingerprint
    return key


def unregister_session(prefix_hash: str) -> None:
    """Drop a process-local registration. Unknown hashes are ignored."""
    _SESSIONS.pop(prefix_hash, None)
    _FINGERPRINTS.pop(prefix_hash, None)
    _FACTORIES.pop(prefix_hash, None)


def set_message_store(store: MessageStore | None) -> None:
    """Install the transcript/output store. ``None`` inlines direct-map results."""
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
    model_settings: dict[str, Any] | None = None
    prefix_ref: str | None = None
    activity_options: ActivityOptions | None = None
    require_references: bool = False


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
    model_settings: ModelSettings | None = None,
    prefix_ref: str | None = None,
) -> QueryActivityInput:
    """Build an activity payload that carries this session's retry policy.

    ``model_settings`` is a per-query override. The worker merges it over the
    registered session's settings. It is not part of the execution fingerprint.
    Values must be JSON-serializable; a client ``timeout`` object is not.
    """
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
        model_settings=dict(model_settings) if model_settings else None,
        prefix_ref=prefix_ref,
        activity_options=session.activity_options,
    )


def activity_retry_policy(payload: QueryActivityInput) -> RetryPolicy:
    """Retry settings override legacy fields individually; ``None`` inherits."""
    options = payload.activity_options
    backoff = (
        options.retry_backoff
        if options is not None and options.retry_backoff is not None
        else payload.retry_backoff_seconds
    )
    maximum_backoff = (
        options.max_backoff if options is not None and options.max_backoff is not None else payload.max_backoff_seconds
    )
    attempts = (
        options.maximum_attempts
        if options is not None and options.maximum_attempts is not None
        else payload.max_attempts
    )
    initial = backoff if backoff > 0 else 0.001
    maximum = maximum_backoff if maximum_backoff > 0 else initial
    return RetryPolicy(
        initial_interval=timedelta(seconds=initial),
        backoff_coefficient=options.backoff_coefficient if options is not None else 2.0,
        maximum_interval=timedelta(seconds=maximum),
        maximum_attempts=max(attempts, 1),
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
    output_ref: str | None = None
    resolved_output_mode: ResolvedOutputMode | None = None


def _permanent(message: str) -> ApplicationError:
    return ApplicationError(message, "permanent", type="permanent", non_retryable=True)


async def _resolve_session(payload: QueryActivityInput) -> Session:
    snapshot: FrozenPrefix | None = None
    if payload.prefix_ref is not None:
        if _PREFIX_STORE is None:
            raise _permanent("prefix_ref was set but this worker has no prefix store")
        try:
            snapshot = FrozenPrefix(**json.loads(await _PREFIX_STORE.get(payload.prefix_ref)))
            snapshot.validate()
            if snapshot.prefix_hash != payload.prefix_hash:
                raise ValueError("prefix_ref does not match the requested prefix_hash")
        except (KeyError, ValueError, TypeError) as exc:
            raise _permanent(f"Invalid or missing frozen prefix: {exc}") from exc
    session = _SESSIONS.get(payload.prefix_hash)
    if session is not None:
        return session
    factory = _FACTORIES.get(payload.prefix_hash)
    if factory is None:
        raise _permanent(f"No session or factory registered for prefix {payload.prefix_hash}")
    if snapshot is None:
        raise _permanent("A registered session factory requires prefix_ref")
    try:
        created = factory(snapshot)
        session = created if isinstance(created, Session) else await created
        if not isinstance(session, Session):
            raise TypeError("Session factory must return a Session")
        actual = compute_prefix_hash(
            output_mode=session.resolved_output_mode,
            instructions=session.instructions,
            rendered_documents=session.rendered_documents,
            tools=session.tool_specs,
        )
        if session.prefix_hash != payload.prefix_hash or actual != payload.prefix_hash:
            raise ValueError("Reconstructed session does not match the frozen prefix hash")
    except (ForkReduceError, ValueError, TypeError) as exc:
        raise _permanent(f"Cannot reconstruct frozen prefix: {exc}") from exc
    return session


async def heartbeat_while(awaitable: Awaitable[Any], heartbeat_timeout: float | None) -> Any:
    """Await work with periodic heartbeats; safe to use outside a Temporal activity.

    The heartbeat task is always cancelled and awaited, including on model
    failures and activity cancellation. No task is started without a timeout.
    """
    if heartbeat_timeout is None or not activity.in_activity():
        return await awaitable

    async def beat() -> None:
        while True:
            activity.heartbeat()
            await asyncio.sleep(min(heartbeat_timeout / 3, 30.0))

    task = asyncio.create_task(beat())
    try:
        return await awaitable
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def execute_query_activity(payload: QueryActivityInput) -> QueryActivityOutput:
    """Run one query attempt. Temporal activities call this through the defn wrapper."""
    if payload.require_references and _STORE is None:
        raise _permanent("Batch activities require a message store for transcript and output references")
    if payload.require_references and payload.history_json is not None:
        raise _permanent("Batch continuations require history_ref, not history_json")
    session = await _resolve_session(payload)
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
    query_settings = cast(ModelSettings, dict(payload.model_settings)) if payload.model_settings else None
    query: Query[Any] = Query(
        tail=payload.tail,
        output=output_type,
        id=payload.query_id,
        history=history,
        meta=dict(payload.meta),
        round=payload.round,
        model_settings=query_settings,
    )
    try:
        session.prepare_query(query)
        options = payload.activity_options
        result = await heartbeat_while(
            session.run_once(query), options.heartbeat_timeout if options is not None else None
        )
    except ForkReduceError as exc:
        raise _permanent(str(exc)) from exc
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
    output_ref: str | None = None
    if isinstance(result.output, BaseModel):
        output_json = result.output.model_dump_json()
        output_name = type(result.output).__name__
        if _STORE is not None and payload.require_references:
            output_ref = await _STORE.put(f"output:{payload.prefix_hash}:{payload.query_id}", output_json.encode())
            output_json = None
    return QueryActivityOutput(
        query_id=result.query_id,
        prefix_hash=result.prefix_hash,
        output_json=output_json,
        output_type=output_name,
        cache_hit_tokens=result.cache_hit_tokens,
        attempts=result.attempts,
        messages_ref=messages_ref,
        messages_json=messages_json,
        output_ref=output_ref,
        resolved_output_mode=result.resolved_output_mode,
    )


@activity.defn(name="forkreduce.run_query")
async def run_query_activity(payload: QueryActivityInput) -> QueryActivityOutput:
    """Temporal activity entrypoint. One attempt; raise to retry."""
    return await execute_query_activity(payload)


@dataclass
class MapQueriesInput:
    """Map configuration. Batch mode returns only transcript/output references."""

    queries: list[QueryActivityInput]
    batch_size: int | None = None
    child_concurrency: int = 4
    activity_options: ActivityOptions | None = None


def activity_timeout_options(options: ActivityOptions | None) -> dict[str, Any]:
    """Convert JSON-native seconds only at the SDK boundary."""
    selected = options if options is not None else ActivityOptions()
    result: dict[str, Any] = {"start_to_close_timeout": timedelta(seconds=selected.start_to_close)}
    if selected.heartbeat_timeout is not None:
        result["heartbeat_timeout"] = timedelta(seconds=selected.heartbeat_timeout)
    if selected.schedule_to_close is not None:
        result["schedule_to_close_timeout"] = timedelta(seconds=selected.schedule_to_close)
    return result


def child_workflow_id(prefix_hash: str, batch_index: int) -> str:
    """Stable per-parent workflow id, including when a workflow retry changes run id."""
    parent_hash = hashlib.sha256(workflow.info().workflow_id.encode()).hexdigest()
    return f"forkreduce:{parent_hash}:{prefix_hash}:batch:{batch_index}"


async def _execute_activity(payload: QueryActivityInput, retry_policy: RetryPolicy) -> QueryActivityOutput:
    return await workflow.execute_activity(
        run_query_activity,
        payload,
        activity_id=activity_id(payload.prefix_hash, payload.query_id),
        retry_policy=retry_policy,
        **activity_timeout_options(payload.activity_options),
    )


@workflow.defn(name="forkreduce.query_batch")
class QueryBatchWorkflow:
    """Execute one bounded batch with no inline transcripts or structured outputs."""

    @workflow.run
    async def run(self, inputs: MapQueriesInput) -> list[QueryActivityOutput]:
        if any(query.history_json is not None for query in inputs.queries):
            raise _permanent("Batch continuations require history_ref, not history_json")
        queries = [
            replace(
                query,
                require_references=True,
                activity_options=inputs.activity_options or query.activity_options,
            )
            for query in inputs.queries
        ]
        outputs = list(
            await asyncio.gather(*(_execute_activity(query, activity_retry_policy(query)) for query in queries))
        )
        if any(
            output.messages_json is not None
            or output.messages_ref is None
            or output.output_json is not None
            or (output.output_type is not None and output.output_ref is None)
            for output in outputs
        ):
            raise _permanent("Batch activities must return transcript and output references")
        return outputs


@workflow.defn(name="forkreduce.map_queries")
class MapQueriesWorkflow:
    """Map queries directly or through bounded, ordered child batches.

    The legacy list input retains direct activity fan-out. A ``batch_size``
    switches to children and requires a shared MessageStore on every worker.
    """

    @workflow.run
    async def run(self, inputs: list[QueryActivityInput] | MapQueriesInput) -> list[QueryActivityOutput]:
        config = MapQueriesInput(queries=inputs) if isinstance(inputs, list) else inputs
        queries = [
            replace(query, activity_options=config.activity_options or query.activity_options)
            for query in config.queries
        ]
        if config.batch_size is None:
            return list(
                await asyncio.gather(*(_execute_activity(query, activity_retry_policy(query)) for query in queries))
            )
        if isinstance(config.batch_size, bool) or not isinstance(config.batch_size, int) or config.batch_size < 1:
            raise _permanent("batch_size must be a positive integer")
        if (
            isinstance(config.child_concurrency, bool)
            or not isinstance(config.child_concurrency, int)
            or config.child_concurrency < 1
        ):
            raise _permanent("child_concurrency must be a positive integer")
        if any(query.history_json is not None for query in queries):
            raise _permanent("Batch continuations require history_ref, not history_json")
        slots = asyncio.Semaphore(config.child_concurrency)

        async def batch(index: int, items: list[QueryActivityInput]) -> list[QueryActivityOutput]:
            async with slots:
                return await workflow.execute_child_workflow(
                    QueryBatchWorkflow.run,
                    MapQueriesInput(queries=items, activity_options=config.activity_options),
                    id=child_workflow_id(items[0].prefix_hash, index),
                )

        batches = await asyncio.gather(
            *(
                batch(index, queries[start : start + config.batch_size])
                for index, start in enumerate(range(0, len(queries), config.batch_size))
            )
        )
        return [output for outputs in batches for output in outputs]
