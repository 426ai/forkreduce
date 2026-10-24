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
from functools import cache
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

    from pydantic import BaseModel, JsonValue, TypeAdapter
    from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
    from pydantic_ai.settings import ModelSettings, merge_model_settings

    from forkreduce.contrib._temporal_specs import OutputSpecRef as OutputSpecRef
    from forkreduce.contrib._temporal_specs import OutputSpecWriter as OutputSpecWriter
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
    from forkreduce.documents import Document, DocumentFormat
    from forkreduce.errors import ForkReduceError
    from forkreduce.failures import Failure
    from forkreduce.models import canonical_value, request_body_settings
    from forkreduce.output_modes import OutputMode, ResolvedOutputMode, UnforcedToolsPolicy
    from forkreduce.output_specs import DynamicOutputs, OutputSpec
    from forkreduce.policy import Policy
    from forkreduce.prefix import QUERY_SEPARATOR, MessageLayout
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
_APP_META_ADAPTER = TypeAdapter(dict[str, JsonValue])
_APP_META_MAX_BYTES = 16 * 1024


def _app_meta_json(app_meta: Mapping[str, Any]) -> str:
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
    app_meta: dict[str, Any] = field(default_factory=dict)
    output_feedback_retries: int = 0
    dynamic_outputs: DynamicOutputs | None = None
    document_format: DocumentFormat = "blocks"
    message_layout: MessageLayout = "split"

    @property
    def query_separator(self) -> str:
        return QUERY_SEPARATOR

    @property
    def dynamic_output_instructions(self) -> bool:
        return True

    @property
    def documents(self) -> list[Document]:
        """Fresh document objects suitable for ``build(documents=...)``."""
        return [Document.model_validate(item) for item in json.loads(self.documents_json)]

    def validate(self) -> None:
        """Reject corrupt snapshots before invoking a worker factory."""
        _app_meta_json(self.app_meta)
        if not isinstance(self.query_separator, str) or not isinstance(self.dynamic_output_instructions, bool):
            raise ValueError("Invalid frozen query layout options")
        if self.document_format not in {"blocks", "raw"} or self.message_layout not in {"split", "single"}:
            raise ValueError("Invalid document format or message layout in frozen prefix")
        if self.dynamic_outputs not in {None, "generic_tool", "prompted"}:
            raise ValueError("Invalid dynamic_outputs in frozen prefix")
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
            document_format=self.document_format,
            message_layout=self.message_layout,
            query_separator=self.query_separator,
        )
        if actual != self.prefix_hash:
            raise ValueError("Frozen prefix hash does not match its contents")
        for item in json.loads(self.documents_json):
            Document.model_validate(item)


@dataclass(frozen=True)
class QueryLayout:
    """Opt-in tail layout configuration; absent from legacy frozen payloads."""

    query_separator: str = QUERY_SEPARATOR
    dynamic_output_instructions: bool = True


@dataclass(frozen=True)
class ConfiguredFrozenPrefix(FrozenPrefix):
    """Nondefault layout snapshot; plain FrozenPrefix keeps its original fields."""

    layout_options: QueryLayout = field(kw_only=True)

    @property
    def query_separator(self) -> str:
        return self.layout_options.query_separator

    @property
    def dynamic_output_instructions(self) -> bool:
        return self.layout_options.dynamic_output_instructions


SessionFactory = Callable[[FrozenPrefix], Session | Awaitable[Session]]
_FACTORIES: dict[str, SessionFactory] = {}
_DEFAULT_FACTORY: SessionFactory | None = None
_PREFIX_STORE: Store | None = None


async def freeze_prefix(
    session: Session,
    store: Store,
    *,
    app_meta: Mapping[str, Any] | None = None,
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
        app_meta=cast(dict[str, Any], json.loads(_app_meta_json({} if app_meta is None else app_meta))),
        output_feedback_retries=session.policy.output_feedback_retries,
        dynamic_outputs=session.dynamic_outputs,
        document_format=session.document_format,
        message_layout=session.message_layout,
    )
    if session.query_separator != QUERY_SEPARATOR or not session.dynamic_output_instructions:
        snapshot = ConfiguredFrozenPrefix(
            **asdict(snapshot),
            layout_options=QueryLayout(session.query_separator, session.dynamic_output_instructions),
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
    if session.dynamic_outputs is not None:
        payload["dynamic_outputs"] = session.dynamic_outputs
    if session.document_format != "blocks":
        payload["document_format"] = session.document_format
    if session.message_layout != "split":
        payload["message_layout"] = session.message_layout
    if session.query_separator != QUERY_SEPARATOR:
        payload["query_separator"] = session.query_separator
    if not session.dynamic_output_instructions:
        payload["dynamic_output_instructions"] = False
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
    output_spec: OutputSpec | OutputSpecRef | None = None


def query_activity_input(
    session: Session,
    *,
    query_id: str,
    tail: str,
    output_name: str | None = None,
    round: int = 0,
    history_ref: str | None = None,
    history_json: str | None = None,
    meta: dict[str, Any] | None = None,
    model_settings: ModelSettings | None = None,
    prefix_ref: str | None = None,
    ref_namespace: str | None = None,
    output_spec: OutputSpec | OutputSpecRef | None = None,
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
    if output_spec is not None:
        if output_name is not None and output_name != output_spec.name:
            raise ValueError("output_name must match output_spec.name")
        if isinstance(output_spec, OutputSpec):
            session.require_output(output_spec)
        elif session.resolved_output_mode == "tools" and session.dynamic_outputs != "generic_tool":
            raise ValueError("OutputSpec references require a frozen generic tool or prompted session")
        output_name = output_spec.name
        output_spec = replace(output_spec)
    if output_name is None:
        raise ValueError("Set output_name or output_spec")
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
        output_spec=output_spec,
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
    synthetic ``batch_failed`` / ``after_query_failed``; arbitrary error text is
    omitted. ``result_meta`` is a JSON copy of the last non-None hook metadata,
    bounded to 16 KiB, replacing rather than merging earlier round metadata.
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
    result_meta: dict[str, Any] | None = None


@dataclass
class AfterQueryInput:
    """Reference-only query result delivered immediately to an application hook."""

    query: QueryActivityInput
    output: QueryActivityOutput
    round: int
    batch_index: int
    parent_workflow_id: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FollowUpQuery:
    """A replacement query with explicit continuation or fresh-history semantics.

    ``continue`` preserves explicit history or inherits the preceding result's
    messages reference. ``fresh`` clears both history fields before execution.
    The prefix remains fixed; a complete query may select a different output
    specification without changing the frozen prefix.
    """

    query: QueryActivityInput
    history: Literal["continue", "fresh"] = "continue"


@dataclass
class AfterQueryDecision:
    """Finish, reject, or repair a query.

    A repair supplies exactly one tail or complete query, retains the prefix,
    and receives ``root_id#r{round}`` automatically. Tails retain the current
    output specification. Complete queries may switch it and inherit missing
    history references unless wrapped in :class:`FollowUpQuery` with ``fresh``.
    ``result_meta`` must be a JSON object fitting in 16 KiB of compact UTF-8 JSON.
    """

    action: Literal["done", "failed", "follow_up"]
    follow_up_tail: str | None = None
    follow_up_query: FollowUpQuery | QueryActivityInput | None = None
    result_meta: dict[str, Any] | None = None


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
            raw = json.loads(
                await store_get(
                    _PREFIX_STORE,
                    payload.prefix_ref,
                    context=_query_store_context("prefix", payload, ref_namespace(payload.meta)),
                )
            )
            if "layout_options" in raw:
                raw["layout_options"] = QueryLayout(**raw["layout_options"])
                snapshot = ConfiguredFrozenPrefix(**raw)
            else:
                snapshot = FrozenPrefix(**raw)
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


def _validate_session_layout(session: Session, snapshot: FrozenPrefix | None) -> None:
    if isinstance(snapshot, ConfiguredFrozenPrefix) and (
        session.query_separator != snapshot.query_separator
        or session.dynamic_output_instructions != snapshot.dynamic_output_instructions
    ):
        raise _permanent("Worker session query layout does not match the frozen configuration")


async def _resolve_session(payload: QueryActivityInput, *, snapshot: FrozenPrefix | None = None) -> Session:
    if snapshot is None:
        snapshot = await _load_prefix(payload)
    session = _SESSIONS.get(payload.prefix_hash)
    if session is not None:
        _validate_session_layout(session, snapshot)
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
            document_format=session.document_format,
            message_layout=session.message_layout,
            query_separator=session.query_separator,
        )
        if session.prefix_hash != payload.prefix_hash or actual != payload.prefix_hash:
            raise ValueError("Reconstructed session does not match the frozen prefix hash")
    except (ForkReduceError, ValueError, TypeError) as exc:
        raise _permanent(f"Cannot reconstruct frozen prefix: {exc}") from exc
    _validate_session_layout(session, snapshot)
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


def _check_spec_namespace(left: str | None, right: str | None) -> None:
    if left and right and left != right:
        raise ApplicationError(
            "Query and output specification reference namespaces do not match",
            "ref_namespace_mismatch",
            _attempt_detail(),
            type="ref_namespace_mismatch",
            non_retryable=True,
        )


async def _resolve_output_spec(payload: QueryActivityInput, namespace: str | None) -> OutputSpec | None:
    reference = payload.output_spec
    if not isinstance(reference, OutputSpecRef):
        return reference
    _check_spec_namespace(namespace, reference.namespace)
    if _STORE is None:
        raise _permanent("OutputSpec references require a message store on this worker")
    try:
        encoded = await store_get(
            _STORE,
            reference.ref,
            context=_query_store_context("output_spec", payload, namespace or reference.namespace),
        )
    except KeyError as exc:
        raise _permanent("OutputSpec reference blob is missing") from exc
    try:
        if hashlib.sha256(encoded).hexdigest() != reference.sha256:
            raise ValueError("OutputSpec reference digest does not match its blob")
        raw = json.loads(encoded)
        _check_spec_namespace(namespace, raw["namespace"])
        _check_spec_namespace(reference.namespace, raw["namespace"])
        if raw["namespace"] != reference.namespace:
            raise ValueError("OutputSpec reference namespace does not match its blob")
        spec = OutputSpec(**raw["spec"])
        if spec.name != reference.name:
            raise ValueError("OutputSpec reference name does not match its blob")
        return spec
    except (KeyError, ValueError, TypeError) as exc:
        raise _permanent("Invalid OutputSpec reference blob or identity") from exc


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
    # Resolve once per attempt; native feedback reuses this local specification.
    spec = await _resolve_output_spec(payload, namespace)
    output_type = spec or session.output_types.get(payload.output_name)
    if output_type is None:
        raise _permanent(f"Output {payload.output_name!r} is not registered on this session")
    if payload.output_spec is not None and payload.output_spec.name != payload.output_name:
        raise _permanent("output_name does not match output_spec.name")
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
    elif isinstance(result.output, dict):
        output_json = json.dumps(result.output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        output_name = payload.output_name
    if output_json is not None and _STORE is not None and payload.require_references:
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
    """Map configuration; failures raise unless explicitly returned as safe records.

    Hooks force reference-only results, including direct maps, and occupy the
    same concurrency slot as every model/repair round. ``max_rounds`` is a
    positive inclusive round index: initial round 0, default permits repair 1.
    Hook options default independently to one attempt and five minutes.
    ``batch_workflow`` names an application workflow accepting MapQueriesInput
    and returning list[QueryActivityOutput]; children retain all hook/scheduling
    settings and receive their zero-based ``batch_index``.
    """

    queries: list[QueryActivityInput]
    batch_size: int | None = None
    child_concurrency: int = 4
    activity_options: ActivityOptions | None = None
    on_query_failure: Literal["raise", "return"] = "raise"
    activity_concurrency: int | None = None
    task_queue: str | None = None
    warmup: Literal["none", "first", "per_batch"] = "none"
    after_query_activity: str | None = None
    after_query_options: ActivityOptions | None = None
    max_rounds: int = 1
    batch_workflow: str | None = None
    batch_index: int = 0
    activity_name: str = "forkreduce.run_query"


def activity_timeout_options(options: ActivityOptions | None) -> dict[str, Any]:
    """Convert JSON-native seconds only at the SDK boundary."""
    selected = options if options is not None else ActivityOptions()
    result: dict[str, Any] = {"start_to_close_timeout": timedelta(seconds=selected.start_to_close)}
    if selected.heartbeat_timeout is not None:
        result["heartbeat_timeout"] = timedelta(seconds=selected.heartbeat_timeout)
    if selected.schedule_to_close is not None:
        result["schedule_to_close_timeout"] = timedelta(seconds=selected.schedule_to_close)
    return result


def plan_batches(queries: list[_T], batch_size: int) -> list[list[_T]]:
    """Split into ordered batches without mutating the input; safe in workflows."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise _permanent("batch_size must be a positive integer")
    return [queries[start : start + batch_size] for start in range(0, len(queries), batch_size)]


def batch_child_id(parent_id: str, prefix_hash: str, index: int) -> str:
    """Stable child id scoped to the parent workflow id, never its run id."""
    parent_hash = hashlib.sha256(parent_id.encode()).hexdigest()
    return f"forkreduce:{parent_hash}:{prefix_hash}:batch:{index}"


def require_query_references(payload: QueryActivityInput) -> QueryActivityInput:
    """Copy a query into reference-only mode, rejecting inline continuation history."""
    if payload.history_json is not None:
        raise _permanent("Batch continuations require history_ref, not history_json")
    return replace(payload, require_references=True)


def validate_query_references(output: QueryActivityOutput) -> None:
    """Reject inline or missing success references; safe failure records are exempt."""
    if output.failure_kind is None and (
        output.messages_json is not None
        or output.messages_ref is None
        or output.output_json is not None
        or (output.output_type is not None and output.output_ref is None)
    ):
        raise _permanent("Batch activities must return transcript and output references")


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


def query_failure(payload: QueryActivityInput, error: ActivityError | ChildWorkflowError) -> QueryActivityOutput:
    """Map an activity or child error to a safe record without exposing its text.

    Synthetic hook/batch failure types are never trusted from application error
    types; orchestration generates them itself. Parent workflow cancellation
    propagates; isolated activity cancellation can be returned safely.
    """
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


async def execute_query(
    payload: QueryActivityInput,
    options: ActivityOptions | None = None,
    *,
    on_query_failure: Literal["raise", "return"] = "raise",
    task_queue: str | None = None,
    require_references: bool = False,
    activity_name: str = "forkreduce.run_query",
) -> QueryActivityOutput:
    """Schedule a query in a workflow, optionally mapping terminal errors safely.

    Explicit options override the payload options. Reference mode rejects inline
    history and validates successful results. Parent cancellation always propagates.
    """
    payload = replace(payload, activity_options=options or payload.activity_options)
    if require_references or payload.require_references:
        payload = require_query_references(payload)
    if workflow.in_workflow():
        parent = workflow.info().parent
        payload = replace(payload, parent_workflow_id=parent.workflow_id if parent is not None else None)
    try:
        output: QueryActivityOutput = await workflow.execute_activity(
            activity_name,
            payload,
            result_type=QueryActivityOutput,
            activity_id=activity_id(payload.prefix_hash, payload.query_id),
            retry_policy=activity_retry_policy(payload),
            task_queue=task_queue,
            **activity_timeout_options(payload.activity_options),
        )
    except ActivityError as exc:
        if on_query_failure == "raise" or (workflow.in_workflow() and workflow.cancellation_reason() is not None):
            raise
        return query_failure(payload, exc)
    if payload.require_references:
        validate_query_references(output)
    return output


def _validate_scheduling(config: MapQueriesInput) -> None:
    limit = config.activity_concurrency
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise _permanent("activity_concurrency must be a positive integer or None")
    if config.warmup not in {"none", "first", "per_batch"}:
        raise _permanent("warmup must be 'none', 'first', or 'per_batch'")
    if isinstance(config.max_rounds, bool) or not isinstance(config.max_rounds, int) or config.max_rounds < 1:
        raise _permanent("max_rounds must be a positive integer")


def _after_query_failure(
    query: QueryActivityInput,
    kind: Literal["permanent", "unknown"],
    result_meta: dict[str, Any] | None,
    *,
    feedback_retries_used: int = 0,
) -> QueryActivityOutput:
    return QueryActivityOutput(
        query.query_id,
        query.prefix_hash,
        None,
        None,
        0,
        0,
        failure_kind=kind,
        failure_message=_FAILURE_MESSAGES[kind],
        failure_type="after_query_failed",
        result_meta=result_meta,
        feedback_retries_used=feedback_retries_used,
        meta={"feedback_retries_used": feedback_retries_used},
    )


def _copy_result_meta(value: dict[str, Any]) -> dict[str, Any]:
    validated = _APP_META_ADAPTER.validate_python(value, strict=True)
    encoded = json.dumps(validated, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > 16 * 1024:
        raise ValueError("result_meta must not exceed 16 KiB of compact UTF-8 JSON")
    return cast(dict[str, Any], json.loads(encoded))


async def _query_pipeline(query: QueryActivityInput, config: MapQueriesInput) -> QueryActivityOutput:
    if config.after_query_activity is None:
        return await execute_query(
            query,
            on_query_failure=config.on_query_failure,
            task_queue=config.task_queue,
            activity_name=config.activity_name,
        )
    current = require_query_references(replace(query, round=0))
    root_id = query.query_id
    result_meta: dict[str, Any] | None = None
    hook_options = config.after_query_options or ActivityOptions()
    # Hook retries are independent of the model/session policy, including when
    # a caller configures only a hook timeout and leaves retry fields unset.
    hook_policy = activity_retry_policy(QueryActivityInput("", "", "", "", activity_options=hook_options))
    info = workflow.info()
    parent_id = info.parent.workflow_id if info.parent is not None else info.workflow_id
    while True:
        output = await execute_query(
            current,
            config.activity_options,
            on_query_failure=config.on_query_failure,
            task_queue=config.task_queue,
            require_references=True,
            activity_name=config.activity_name,
        )
        output = replace(output, result_meta=result_meta)
        try:
            decision: AfterQueryDecision = await workflow.execute_activity(
                config.after_query_activity,
                AfterQueryInput(current, output, current.round, config.batch_index, parent_id, dict(current.meta)),
                result_type=AfterQueryDecision,
                activity_id=f"{activity_id(current.prefix_hash, current.query_id)}:after_query:r{current.round}",
                task_queue=config.task_queue,
                retry_policy=hook_policy,
                **activity_timeout_options(hook_options),
            )
        except ActivityError:
            if config.on_query_failure == "raise" or (
                workflow.in_workflow() and workflow.cancellation_reason() is not None
            ):
                raise
            return _after_query_failure(
                current, "unknown", result_meta, feedback_retries_used=output.feedback_retries_used
            )
        try:
            if decision.result_meta is not None:
                result_meta = _copy_result_meta(decision.result_meta)
            if decision.action == "done":
                return replace(output, result_meta=result_meta)
            if decision.action == "failed" or current.round >= config.max_rounds:
                return _after_query_failure(
                    current, "permanent", result_meta, feedback_retries_used=output.feedback_retries_used
                )
            if decision.action != "follow_up" or (
                (decision.follow_up_tail is None) == (decision.follow_up_query is None)
            ):
                raise ValueError("follow_up requires exactly one tail or query")
            next_query = decision.follow_up_query
            history = "continue"
            if isinstance(next_query, FollowUpQuery):
                history = next_query.history
                if history not in {"continue", "fresh"}:
                    raise ValueError("follow_up history must be 'continue' or 'fresh'")
                if not isinstance(next_query.query, QueryActivityInput):
                    raise TypeError("follow_up_query must contain a QueryActivityInput")
                next_query = next_query.query
            if next_query is None:
                next_query = replace(current, tail=cast(str, decision.follow_up_tail), history_ref=output.messages_ref)
            elif not isinstance(next_query, QueryActivityInput):
                raise TypeError("follow_up_query must contain a QueryActivityInput")
            elif next_query.prefix_hash != current.prefix_hash:
                raise ValueError("follow_up must retain the same prefix_hash")
            elif history == "fresh":
                next_query = replace(next_query, history_ref=None, history_json=None)
            elif next_query.history_ref is None:
                next_query = replace(next_query, history_ref=output.messages_ref)
            current = require_query_references(
                replace(next_query, query_id=f"{root_id}#r{current.round + 1}", round=current.round + 1)
            )
        except (ValueError, TypeError, ApplicationError):
            return _after_query_failure(
                current, "permanent", result_meta, feedback_retries_used=output.feedback_retries_used
            )


async def _run_queries(queries: list[QueryActivityInput], config: MapQueriesInput) -> list[QueryActivityOutput]:
    """Hold a slot through each query; warmup gates only the selected scope."""
    _validate_scheduling(config)
    slots = asyncio.Semaphore(config.activity_concurrency) if config.activity_concurrency is not None else None

    async def execute(query: QueryActivityInput) -> QueryActivityOutput:
        return await _query_pipeline(query, config)

    async def bounded(query: QueryActivityInput) -> QueryActivityOutput:
        if slots is None:
            return await execute(query)
        async with slots:
            return await execute(query)

    first: list[QueryActivityOutput] = []
    if config.warmup != "none" and queries:
        first.append(await bounded(queries[0]))
        queries = queries[1:]
    return [*first, *await asyncio.gather(*(bounded(query) for query in queries))]


@workflow.defn(name="forkreduce.query_batch")
class QueryBatchWorkflow:
    """Execute one bounded batch with no inline transcripts or structured outputs."""

    @workflow.run
    async def run(self, inputs: MapQueriesInput) -> list[QueryActivityOutput]:
        queries = [
            require_query_references(replace(query, activity_options=inputs.activity_options or query.activity_options))
            for query in inputs.queries
        ]
        return await _run_queries(queries, inputs)


@workflow.defn(name="forkreduce.map_queries")
class MapQueriesWorkflow:
    """Map queries directly or through bounded, ordered child batches.

    The legacy list input retains direct activity fan-out. A ``batch_size``
    switches to children and requires a shared MessageStore on every worker.
    """

    @workflow.run
    async def run(self, inputs: list[QueryActivityInput] | MapQueriesInput) -> list[QueryActivityOutput]:
        config = MapQueriesInput(queries=inputs) if isinstance(inputs, list) else inputs
        _validate_scheduling(config)
        queries = [
            replace(query, activity_options=config.activity_options or query.activity_options)
            for query in config.queries
        ]
        if config.batch_size is None:
            return await _run_queries(queries, config)
        if isinstance(config.batch_size, bool) or not isinstance(config.batch_size, int) or config.batch_size < 1:
            raise _permanent("batch_size must be a positive integer")
        if (
            isinstance(config.child_concurrency, bool)
            or not isinstance(config.child_concurrency, int)
            or config.child_concurrency < 1
        ):
            raise _permanent("child_concurrency must be a positive integer")
        queries = [require_query_references(query) for query in queries]
        slots = asyncio.Semaphore(config.child_concurrency)

        async def batch(index: int, items: list[QueryActivityInput]) -> list[QueryActivityOutput]:
            async with slots:
                try:
                    outputs: list[QueryActivityOutput] = await workflow.execute_child_workflow(
                        config.batch_workflow or "forkreduce.query_batch",
                        replace(
                            config,
                            queries=items,
                            batch_size=None,
                            batch_index=index,
                            warmup="per_batch" if config.warmup == "per_batch" else "none",
                        ),
                        id=batch_child_id(workflow.info().workflow_id, items[0].prefix_hash, index),
                        result_type=list[QueryActivityOutput],
                        task_queue=config.task_queue,
                    )
                    return outputs
                except ChildWorkflowError as exc:
                    if config.on_query_failure == "raise" or (
                        workflow.in_workflow() and workflow.cancellation_reason() is not None
                    ):
                        raise
                    return [query_failure(query, exc) for query in items]

        first: list[QueryActivityOutput] = []
        batch_offset = 0
        if config.warmup == "first" and queries:
            first = await batch(0, queries[:1])
            queries = queries[1:]
            batch_offset = 1

        batches = await asyncio.gather(
            *(
                batch(index + batch_offset, items)
                for index, items in enumerate(plan_batches(queries, config.batch_size))
            )
        )
        return [*first, *(output for outputs in batches for output in outputs)]


@dataclass(frozen=True)
class TemporalNames:
    """Serializable registration names; definitions are generated only on workers.

    Only ``prefix`` is data. Register ``activities`` and ``workflows`` on a
    worker, then start maps by ``map_workflow_name``. An empty prefix preserves
    the original activity and workflow definitions exactly.
    """

    prefix: str = ""

    @property
    def activity_name(self) -> str:
        return self.prefix + "forkreduce.run_query"

    @property
    def map_workflow_name(self) -> str:
        return self.prefix + "forkreduce.map_queries"

    @property
    def batch_workflow_name(self) -> str:
        return self.prefix + "forkreduce.query_batch"

    @property
    def activities(self) -> tuple[Callable[[QueryActivityInput], Awaitable[QueryActivityOutput]], ...]:
        return (_temporal_definitions(self.prefix)[0],)

    @property
    def workflows(self) -> tuple[type[MapQueriesWorkflow], type[QueryBatchWorkflow]]:
        definitions = _temporal_definitions(self.prefix)
        return definitions[1], definitions[2]


def forkreduce_temporal_names(*, prefix: str = "") -> TemporalNames:
    """Choose a registration namespace without changing query or child IDs."""
    return TemporalNames(prefix=prefix)


@cache
def _temporal_definitions(
    prefix: str,
) -> tuple[
    Callable[[QueryActivityInput], Awaitable[QueryActivityOutput]],
    type[MapQueriesWorkflow],
    type[QueryBatchWorkflow],
]:
    if not prefix:
        return run_query_activity, MapQueriesWorkflow, QueryBatchWorkflow
    names = TemporalNames(prefix)

    @activity.defn(name=names.activity_name)
    async def named_activity(payload: QueryActivityInput) -> QueryActivityOutput:
        return await execute_query_activity(payload)

    class NamedMapQueriesWorkflow(MapQueriesWorkflow):
        async def run(self, inputs: list[QueryActivityInput] | MapQueriesInput) -> list[QueryActivityOutput]:
            config = MapQueriesInput(queries=inputs) if isinstance(inputs, list) else inputs
            return await super().run(
                replace(
                    config,
                    activity_name=names.activity_name,
                    batch_workflow=(
                        config.batch_workflow if config.batch_workflow is not None else names.batch_workflow_name
                    ),
                )
            )

    class NamedQueryBatchWorkflow(QueryBatchWorkflow):
        async def run(self, inputs: MapQueriesInput) -> list[QueryActivityOutput]:
            return await super().run(replace(inputs, activity_name=names.activity_name))

    # Temporal's sandbox imports workflow classes by module and Python name.
    # Export cached worker-local definitions for passthrough sandbox imports.
    suffix = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    NamedMapQueriesWorkflow.__name__ = f"_ForkReduceMapQueries_{suffix}"
    NamedMapQueriesWorkflow.__qualname__ = NamedMapQueriesWorkflow.__name__
    NamedMapQueriesWorkflow.run.__qualname__ = f"{NamedMapQueriesWorkflow.__name__}.run"
    NamedQueryBatchWorkflow.__name__ = f"_ForkReduceQueryBatch_{suffix}"
    NamedQueryBatchWorkflow.__qualname__ = NamedQueryBatchWorkflow.__name__
    NamedQueryBatchWorkflow.run.__qualname__ = f"{NamedQueryBatchWorkflow.__name__}.run"
    workflow.run(NamedMapQueriesWorkflow.run)
    workflow.run(NamedQueryBatchWorkflow.run)
    globals()[NamedMapQueriesWorkflow.__name__] = NamedMapQueriesWorkflow
    globals()[NamedQueryBatchWorkflow.__name__] = NamedQueryBatchWorkflow
    return (
        named_activity,
        workflow.defn(name=names.map_workflow_name)(NamedMapQueriesWorkflow),
        workflow.defn(name=names.batch_workflow_name)(NamedQueryBatchWorkflow),
    )
