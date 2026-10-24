"""Temporal adapter.

Each query is one activity. The idempotency key is ``prefix_hash + query_id``,
exposed as :func:`activity_id`. The activity performs a single model attempt and
raises :class:`temporalio.exceptions.ApplicationError` with ``non_retryable``
set from the failure class, so Temporal's retry policy is the bound.

Register a worker-owned session with :func:`register_session`, or persist its
prefix using :func:`freeze_prefix` and configure :func:`register_session_factory`
or :func:`set_default_session_factory`. Factories reconstruct the model and output
classes locally; neither is serialized.

Transcripts use :class:`MessageStore` references when configured. Direct maps
retain inline results for compatibility; child batches require references for
successful transcripts and structured outputs. ``MapQueriesInput.on_query_failure``
can return safe failure records instead of failing the map; cancellation of the
workflow always propagates. Direct callers can also request reference-only results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from typing import Any, Literal, TypeVar, cast

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    ChildWorkflowError,
    RetryState,
    TimeoutError,
    is_cancelled_exception,
)

with workflow.unsafe.imports_passed_through():
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic import BaseModel, TypeAdapter
    from pydantic import JsonValue as JSONValue
    from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
    from pydantic_ai.settings import ModelSettings, merge_model_settings

    from forkreduce.contrib._temporal_store import (
        ContextMessageStore as ContextMessageStore,
    )
    from forkreduce.contrib._temporal_store import (
        MessageStore as MessageStore,
    )
    from forkreduce.contrib._temporal_store import (
        PrefixStore as PrefixStore,
    )
    from forkreduce.contrib._temporal_store import (
        Store as Store,
    )
    from forkreduce.contrib._temporal_store import (
        StoreContext as StoreContext,
    )
    from forkreduce.contrib._temporal_store import (
        StoreKind,
        ref_namespace,
        store_context,
        store_get,
        store_put,
    )
    from forkreduce.contrib._temporal_store import (
        delete_ref as delete_ref,
    )
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

_T = TypeVar("_T")


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

    async def delete(self, ref: str, *, context: StoreContext | None = None) -> None:
        """Explicit application-owned cleanup; unknown references are ignored."""
        self._data.pop(ref, None)


_SESSIONS: dict[str, Session] = {}
_FINGERPRINTS: dict[str, str] = {}
_STORE: Store | None = None
_APP_META_ADAPTER = TypeAdapter(dict[str, JSONValue])
_APP_META_MAX_BYTES = 16 * 1024


def _app_meta_json(app_meta: Mapping[str, JSONValue]) -> str:
    if not isinstance(app_meta, Mapping):
        raise ValueError("Frozen prefix app_meta must be a JSON object")
    validated = _APP_META_ADAPTER.validate_python(dict(app_meta), strict=True)
    encoded = json.dumps(validated, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > _APP_META_MAX_BYTES:
        raise ValueError("Frozen prefix app_meta must not exceed 16 KiB of compact UTF-8 JSON")
    return encoded


@dataclass(frozen=True)
class FrozenPrefix:
    """JSON-only snapshot; worker factories supply models and output classes.

    ``documents_json`` preserves cleaned document records. ``rendered_documents``
    preserves the exact prefix, including any custom metadata rendering.
    A factory must reproduce custom metadata rendering as well as the model and
    output schemas: the reconstructed session's complete prefix is verified.

    ``app_meta`` is application-owned reconstruction metadata, not prompt content.
    It must fit in 16 KiB of compact UTF-8 JSON and never changes prefix or
    execution identity. Older snapshots without it receive an independent dict.
    """

    prefix_hash: str
    instructions: str
    documents_json: str
    rendered_documents: str
    output_mode: OutputMode
    tools_json: str
    resolved_output_mode: ResolvedOutputMode | None = None
    on_unforced_tools: UnforcedToolsPolicy = "warn"
    app_meta: dict[str, JSONValue] = field(default_factory=dict)
    output_feedback_retries: int = 0

    @property
    def documents(self) -> list[Document]:
        """Fresh document objects suitable for ``build(documents=...)``."""
        return [Document.model_validate(item) for item in json.loads(self.documents_json)]

    def validate(self) -> None:
        """Reject corrupt snapshots before invoking a worker factory."""
        _app_meta_json(self.app_meta)
        if (
            isinstance(self.output_feedback_retries, bool)
            or not isinstance(self.output_feedback_retries, int)
            or self.output_feedback_retries < 0
        ):
            raise ValueError("Frozen output_feedback_retries must be a non-negative integer")
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
_DEFAULT_FACTORY: SessionFactory | None = None
_PREFIX_STORE: Store | None = None


async def freeze_prefix(
    session: Session,
    store: Store,
    *,
    app_meta: Mapping[str, JSONValue] | None = None,
) -> str:
    """Persist a prefix without serializing the model or Python output classes.

    Optional ``app_meta`` is JSON-copied for the worker factory and limited to
    16 KiB of compact UTF-8 JSON. It does not affect the prefix hash or execution
    fingerprint. Store only application metadata safe to persist with the prefix.
    """
    snapshot = FrozenPrefix(
        prefix_hash=session.prefix_hash,
        instructions=session.instructions,
        documents_json=json.dumps([doc.model_dump(mode="json") for doc in session.documents.documents]),
        rendered_documents=session.rendered_documents,
        output_mode=session.output_mode,
        tools_json=json.dumps(session.tool_specs),
        resolved_output_mode=session.resolved_output_mode,
        on_unforced_tools=session.on_unforced_tools,
        app_meta=cast(dict[str, JSONValue], json.loads(_app_meta_json({} if app_meta is None else app_meta))),
        output_feedback_retries=session.policy.output_feedback_retries,
    )
    snapshot.validate()
    encoded = json.dumps(asdict(snapshot), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return await store_put(
        store,
        f"prefix:{session.prefix_hash}",
        encoded,
        context=store_context("prefix", session.prefix_hash, namespace=ref_namespace(snapshot.app_meta)),
    )


def set_prefix_store(store: Store | None) -> None:
    """Install the worker store used to resolve ``QueryActivityInput.prefix_ref``."""
    global _PREFIX_STORE
    _PREFIX_STORE = store


def register_session_factory(prefix_hash: str, factory: SessionFactory) -> None:
    """Register a sync or async factory for one exact prefix hash.

    Unless a session is explicitly registered, the factory runs on every activity
    attempt. The adapter does not implicitly share session budgets across those
    attempts or unrelated workflows. A factory can cache its Session or call
    :func:`register_session` when worker-wide reuse and budget sharing are wanted.
    """
    _FACTORIES[prefix_hash] = factory


def set_default_session_factory(factory: SessionFactory | None) -> None:
    """Set the fallback factory for prefixes without an exact registration.

    Registered sessions take precedence over all factories. An exact factory
    takes precedence over this default; passing ``None`` removes the default.
    Factories run on every activity attempt unless a session is registered.
    Missing factories raise retryable ``session_factory_missing`` errors so
    workers can acquire configuration before a later attempt.
    """
    global _DEFAULT_FACTORY
    _DEFAULT_FACTORY = factory


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
    if policy.output_feedback_retries:
        payload["output_feedback_retries"] = policy.output_feedback_retries
    if session.output_validator is not None:
        payload["output_validator"] = canonical_value(session.output_validator)
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


def set_message_store(store: Store | None) -> None:
    """Install the transcript/output store. ``None`` inlines direct-map results."""
    global _STORE
    _STORE = store


def activity_id(prefix_hash: str, query_id: str) -> str:
    """Idempotency key for one query on one frozen prefix."""
    return f"forkreduce:{prefix_hash}:{query_id}"


def _attempt_detail() -> dict[str, int]:
    return {"forkreduce_attempt": activity.info().attempt if activity.in_activity() else 1}


def to_application_error(failure: Failure, *, feedback_retries_used: int = 0) -> ApplicationError:
    """Map a classified failure onto a Temporal application error.

    Validation and transport failures stay retryable. Permanent failures and
    budget exhaustion are ``non_retryable``.
    """
    detail = _attempt_detail()
    if feedback_retries_used:
        detail["feedback_retries_used"] = feedback_retries_used
    return ApplicationError(
        failure.message,
        failure.kind.value,
        detail,
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
    parent_workflow_id: str | None = None


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
    ref_namespace: str | None = None,
) -> QueryActivityInput:
    """Build an activity payload that carries this session's retry policy.

    ``model_settings`` is a per-query override. The worker merges it over the
    registered session's settings. It is not part of the execution fingerprint.
    Values must be JSON-serializable; a client ``timeout`` object is not.

    ``ref_namespace`` routes references without changing the prefix identity.
    A non-``None`` value overrides the copied ``meta`` namespace; otherwise the
    metadata value is preserved. Conflicting query and snapshot namespaces fail
    before session reconstruction or model execution.
    """
    policy = session.policy
    query_meta = dict(meta or {})
    if ref_namespace is not None:
        query_meta["ref_namespace"] = ref_namespace
    return QueryActivityInput(
        prefix_hash=session.prefix_hash,
        query_id=query_id,
        tail=tail,
        output_name=output_name,
        round=round,
        history_ref=history_ref,
        history_json=history_json,
        meta=query_meta,
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


QueryFailureKind = Literal["retryable_exhausted", "permanent", "budget", "timeout", "cancelled", "unknown"]
"""Stable terminal query categories; ``None`` on the result denotes success."""


@dataclass
class QueryActivityOutput:
    """Activity result, or a safe terminal failure without bodies or references.

    For failures, ``attempts`` counts observed activity executions, or zero if
    unknown (including child failures). Successes retain Session.run_once counts.
    ``failure_type`` contains only a whitelisted raw application type, or the
    synthetic ``batch_failed`` for a child failure; arbitrary error text is omitted.
    """

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
    failure_kind: QueryFailureKind | None = None
    failure_message: str | None = None
    failure_type: str | None = None
    feedback_retries_used: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def _permanent(message: str) -> ApplicationError:
    return ApplicationError(message, "permanent", _attempt_detail(), type="permanent", non_retryable=True)


def _query_store_context(kind: StoreKind, payload: QueryActivityInput, namespace: str | None) -> StoreContext:
    return store_context(
        kind,
        payload.prefix_hash,
        namespace=namespace,
        query_id=payload.query_id,
        round=payload.round,
        parent_workflow_id=payload.parent_workflow_id,
    )


async def _load_prefix(payload: QueryActivityInput) -> FrozenPrefix | None:
    """Read with query metadata before snapshot routing metadata is available."""
    snapshot: FrozenPrefix | None = None
    if payload.prefix_ref is not None:
        if _PREFIX_STORE is None:
            raise _permanent("prefix_ref was set but this worker has no prefix store")
        try:
            snapshot = FrozenPrefix(
                **json.loads(
                    await store_get(
                        _PREFIX_STORE,
                        payload.prefix_ref,
                        context=_query_store_context("prefix", payload, ref_namespace(payload.meta)),
                    )
                )
            )
            snapshot.validate()
            if snapshot.prefix_hash != payload.prefix_hash:
                raise ValueError("prefix_ref does not match the requested prefix_hash")
        except (KeyError, ValueError, TypeError) as exc:
            raise _permanent(f"Invalid or missing frozen prefix: {exc}") from exc
        query_namespace = ref_namespace(payload.meta)
        snapshot_namespace = ref_namespace(snapshot.app_meta)
        if query_namespace and snapshot_namespace and query_namespace != snapshot_namespace:
            raise ApplicationError(
                "Query and frozen prefix reference namespaces do not match",
                "ref_namespace_mismatch",
                _attempt_detail(),
                type="ref_namespace_mismatch",
                non_retryable=True,
            )
    return snapshot


async def _resolve_session(payload: QueryActivityInput, *, snapshot: FrozenPrefix | None = None) -> Session:
    if snapshot is None:
        snapshot = await _load_prefix(payload)
    session = _SESSIONS.get(payload.prefix_hash)
    if session is not None:
        return session
    factory = _FACTORIES.get(payload.prefix_hash)
    if factory is None:
        factory = _DEFAULT_FACTORY
    if factory is None:
        raise ApplicationError(
            f"No session or factory registered for prefix {payload.prefix_hash}",
            "session_factory_missing",
            _attempt_detail(),
            type="session_factory_missing",
        )
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


async def heartbeat_while(awaitable: Awaitable[_T], heartbeat_timeout: float | None) -> _T:
    """Await work with periodic heartbeats; safe to use outside a Temporal activity.

    The heartbeat task is always cancelled and awaited, including on work
    failures and activity cancellation. No task is started without a timeout.
    """
    if heartbeat_timeout is None or not activity.in_activity():
        return await awaitable

    async def beat() -> None:
        while True:
            activity.heartbeat(_attempt_detail())
            await asyncio.sleep(min(heartbeat_timeout / 3, 30.0))

    task = asyncio.create_task(beat())
    try:
        return await awaitable
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def execute_query_activity(payload: QueryActivityInput) -> QueryActivityOutput:
    """Run one attempt, heartbeating across reference I/O, factory, and model awaits."""
    options = payload.activity_options
    return await heartbeat_while(
        _execute_query_activity(payload), options.heartbeat_timeout if options is not None else None
    )


async def _execute_query_activity(payload: QueryActivityInput) -> QueryActivityOutput:
    """Activity work inside the single outer heartbeat lifetime."""
    if payload.require_references and _STORE is None:
        raise _permanent("Batch activities require a message store for transcript and output references")
    if payload.require_references and payload.history_json is not None:
        raise _permanent("Batch continuations require history_ref, not history_json")
    snapshot = await _load_prefix(payload)
    namespace = (
        ref_namespace(payload.meta)
        if "ref_namespace" in payload.meta
        else ref_namespace(snapshot.app_meta)
        if snapshot is not None
        else None
    )
    session = await _resolve_session(payload, snapshot=snapshot)
    output_type = session.output_types.get(payload.output_name)
    if output_type is None:
        raise _permanent(f"Output {payload.output_name!r} is not registered on this session")
    history: list[ModelMessage] | None = None
    if payload.history_ref is not None and payload.history_json is not None:
        raise _permanent("Set only one of history_ref and history_json")
    if payload.history_json is not None:
        history = list(ModelMessagesTypeAdapter.validate_json(payload.history_json))
    elif payload.history_ref is not None:
        if _STORE is None:
            raise _permanent("history_ref was set but this worker has no message store")
        history = list(
            ModelMessagesTypeAdapter.validate_json(
                await store_get(
                    _STORE, payload.history_ref, context=_query_store_context("history", payload, namespace)
                )
            )
        )
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
        result = await session.run_once(query)
    except ForkReduceError as exc:
        raise _permanent(str(exc)) from exc
    if result.failure is not None:
        raise to_application_error(result.failure, feedback_retries_used=result.feedback_retries_used)
    dumped = ModelMessagesTypeAdapter.dump_json(result.messages)
    messages_ref: str | None = None
    messages_json: str | None = None
    if _STORE is not None:
        messages_ref = await store_put(
            _STORE,
            f"{payload.prefix_hash}:{payload.query_id}",
            dumped,
            context=_query_store_context("messages", payload, namespace),
        )
    else:
        messages_json = dumped.decode()
    output_json: str | None = None
    output_name: str | None = None
    output_ref: str | None = None
    if isinstance(result.output, BaseModel):
        output_json = result.output.model_dump_json()
        output_name = type(result.output).__name__
        if _STORE is not None and payload.require_references:
            output_ref = await store_put(
                _STORE,
                f"output:{payload.prefix_hash}:{payload.query_id}",
                output_json.encode(),
                context=_query_store_context("output", payload, namespace),
            )
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
        feedback_retries_used=result.feedback_retries_used,
        meta={"feedback_retries_used": result.feedback_retries_used},
    )


@activity.defn(name="forkreduce.run_query")
async def run_query_activity(payload: QueryActivityInput) -> QueryActivityOutput:
    """Temporal activity entrypoint. One attempt; raise to retry."""
    return await execute_query_activity(payload)


@dataclass
class MapQueriesInput:
    """Map configuration; failures raise unless explicitly returned as safe records."""

    queries: list[QueryActivityInput]
    batch_size: int | None = None
    child_concurrency: int = 4
    activity_options: ActivityOptions | None = None
    on_query_failure: Literal["raise", "return"] = "raise"


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


_FAILURE_MESSAGES: dict[QueryFailureKind, str] = {
    "retryable_exhausted": "Query failed after retry attempts were exhausted.",
    "permanent": "Query failed permanently.",
    "budget": "Query exceeded its execution budget.",
    "timeout": "Query activity timed out.",
    "cancelled": "Query activity was cancelled.",
    "unknown": "Query failed without a query-level failure classification.",
}
_FAILURE_TYPES = frozenset(
    {
        "validation",
        "transport",
        "permanent",
        "budget",
        "session_factory_missing",
        "ref_namespace_mismatch",
    }
)


def _failure_output(payload: QueryActivityInput, error: ActivityError | ChildWorkflowError) -> QueryActivityOutput:
    cause = error.__cause__
    attempts = 0
    feedback_retries_used = 0
    details = (
        ()
        if isinstance(error, ChildWorkflowError)
        else cause.details
        if isinstance(cause, ApplicationError)
        else cause.last_heartbeat_details
        if isinstance(cause, TimeoutError)
        else ()
    )
    for detail in details:
        if isinstance(detail, dict):
            attempt = detail.get("forkreduce_attempt")
            if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0:
                attempts = attempt
            feedback = detail.get("feedback_retries_used")
            if isinstance(feedback, int) and not isinstance(feedback, bool) and feedback >= 0:
                feedback_retries_used = feedback
    kind: QueryFailureKind
    if isinstance(error, ChildWorkflowError):
        kind = "unknown"
    elif is_cancelled_exception(error):
        kind = "cancelled"
    elif isinstance(cause, TimeoutError) or error.retry_state == RetryState.TIMEOUT:
        kind = "timeout"
    elif isinstance(cause, ApplicationError):
        if cause.type == "budget":
            kind = "budget"
        elif cause.non_retryable:
            kind = "permanent"
        elif error.retry_state == RetryState.MAXIMUM_ATTEMPTS_REACHED:
            kind = "retryable_exhausted"
        else:
            kind = "unknown"
    elif error.retry_state == RetryState.MAXIMUM_ATTEMPTS_REACHED:
        kind = "retryable_exhausted"
    else:
        kind = "unknown"
    return QueryActivityOutput(
        query_id=payload.query_id,
        prefix_hash=payload.prefix_hash,
        output_json=None,
        output_type=None,
        cache_hit_tokens=0,
        attempts=attempts,
        failure_kind=kind,
        failure_message=_FAILURE_MESSAGES[kind],
        failure_type=(
            "batch_failed"
            if isinstance(error, ChildWorkflowError)
            else cause.type
            if isinstance(cause, ApplicationError) and cause.type in _FAILURE_TYPES
            else None
        ),
        feedback_retries_used=feedback_retries_used,
        meta={"feedback_retries_used": feedback_retries_used},
    )


async def _execute_activity(
    payload: QueryActivityInput,
    retry_policy: RetryPolicy,
    on_query_failure: Literal["raise", "return"] = "raise",
) -> QueryActivityOutput:
    if workflow.in_workflow():
        parent = workflow.info().parent
        payload = replace(payload, parent_workflow_id=parent.workflow_id if parent is not None else None)
    try:
        return await workflow.execute_activity(
            run_query_activity,
            payload,
            activity_id=activity_id(payload.prefix_hash, payload.query_id),
            retry_policy=retry_policy,
            **activity_timeout_options(payload.activity_options),
        )
    except ActivityError as exc:
        if on_query_failure == "raise" or (workflow.in_workflow() and workflow.cancellation_reason() is not None):
            raise
        return _failure_output(payload, exc)


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
            await asyncio.gather(
                *(_execute_activity(query, activity_retry_policy(query), inputs.on_query_failure) for query in queries)
            )
        )
        if any(
            output.messages_json is not None
            or output.messages_ref is None
            or output.output_json is not None
            or (output.output_type is not None and output.output_ref is None)
            for output in outputs
            if output.failure_kind is None
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
                await asyncio.gather(
                    *(
                        _execute_activity(query, activity_retry_policy(query), config.on_query_failure)
                        for query in queries
                    )
                )
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
                try:
                    return await workflow.execute_child_workflow(
                        QueryBatchWorkflow.run,
                        MapQueriesInput(
                            queries=items,
                            activity_options=config.activity_options,
                            on_query_failure=config.on_query_failure,
                        ),
                        id=child_workflow_id(items[0].prefix_hash, index),
                    )
                except ChildWorkflowError as exc:
                    if config.on_query_failure == "raise" or (
                        workflow.in_workflow() and workflow.cancellation_reason() is not None
                    ):
                        raise
                    return [_failure_output(query, exc) for query in items]

        batches = await asyncio.gather(
            *(
                batch(index, queries[start : start + config.batch_size])
                for index, start in enumerate(range(0, len(queries), config.batch_size))
            )
        )
        return [output for outputs in batches for output in outputs]
