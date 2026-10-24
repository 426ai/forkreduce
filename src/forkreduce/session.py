"""Sessions: a frozen prefix and the queries that fork off it."""

from __future__ import annotations

import copy
import inspect
import json
import time
from collections.abc import Sequence
from typing import Any, TypeVar, cast

from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput, ToolOutput, capture_run_messages
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.usage import RunUsage, UsageLimits

from forkreduce.context import (
    ContextWarning,
    emit_context_warning,
    estimate_tokens,
    prefix_context_warning,
    prefix_text,
    request_context_warning,
    request_text,
)
from forkreduce.documents import DocumentIndex, DocumentsInput, MetaRenderer, coerce_documents
from forkreduce.errors import ForkReduceError, OutputModeConflictError, UnregisteredOutputError
from forkreduce.executor import BudgetLedger, Executor, LocalExecutor, budget_action_failure
from forkreduce.failures import Failure, classify_exception, wrong_output_failure
from forkreduce.output_modes import (
    OutputMode,
    ResolvedOutputMode,
    UnforcedToolsPolicy,
    diagnose_unforced_tools,
    effective_model_settings,
    forced_tools_available,
    resolve_output_mode,
)
from forkreduce.policy import Policy
from forkreduce.prefix import (
    history_matches_prefix,
    prefix_hash,
    prefix_messages,
    warn_if_prefix_mismatch,
)
from forkreduce.query import Query, QueryResult
from forkreduce.stats import cache_hit_tokens, usage_from_messages
from forkreduce.temporal_options import ActivityOptions

T = TypeVar("T")

_REQUEST_LIMIT = UsageLimits(request_limit=1)


class Session:
    """A frozen prefix plus the registered output tools.

    Build one with :func:`forkreduce.build`. Queries share ``instructions``,
    the rendered documents, and (in ``tools`` mode) the whole output tool set.
    The per-query text is only the tail.
    ``activity_options`` configures only the optional Temporal adapter; duration
    values are seconds and do not change local execution or prefix identity.
    """

    def __init__(
        self,
        *,
        model: Model,
        instructions: str,
        documents: DocumentIndex,
        output_types: dict[str, type[BaseModel]],
        output_mode: OutputMode,
        resolved_output_mode: ResolvedOutputMode,
        on_unforced_tools: UnforcedToolsPolicy,
        policy: Policy,
        warmup: bool,
        prefix_hash_value: str,
        tool_specs: list[dict[str, Any]],
        executor: Executor | None,
        model_settings: ModelSettings | None = None,
        activity_options: ActivityOptions | None = None,
    ) -> None:
        self.model = model
        copied = cast(ModelSettings, dict(model_settings)) if model_settings else None
        self.model_settings: ModelSettings | None = copied
        self.instructions = instructions
        self.documents = documents
        self.rendered_documents = documents.rendered
        self.output_types = dict(output_types)
        self.output_mode: OutputMode = output_mode
        self.resolved_output_mode: ResolvedOutputMode = resolved_output_mode
        self.on_unforced_tools: UnforcedToolsPolicy = on_unforced_tools
        self._forced_tools_available = forced_tools_available(model, effective_model_settings(model, copied))
        self.policy = policy
        self.activity_options = activity_options
        self.warmup = warmup
        self.prefix_hash = prefix_hash_value
        self.tool_specs = tool_specs
        self._executor: Executor = executor if executor is not None else LocalExecutor(policy)
        self._budget = BudgetLedger(policy)
        self._warmed = False
        self._ids = 0
        self.context_warning: ContextWarning | None = None
        self._agent: Agent[None, Any] = _build_agent(
            model,
            instructions=instructions,
            output_types=self.output_types,
            output_mode=resolved_output_mode,
            descriptions={
                name: str(spec.get("description", "")) for name, spec in zip(output_types, tool_specs, strict=True)
            },
        )

    @property
    def output_names(self) -> list[str]:
        """Names of the output types registered on this session, in build order."""
        return list(self.output_types)

    def require_output(self, output: type[Any]) -> str:
        """Return the registered tool name or raise :class:`UnregisteredOutputError`."""
        for name, registered in self.output_types.items():
            if registered is output:
                return name
        qualified = getattr(output, "__name__", repr(output))
        names = ", ".join(self.output_names) or "(none)"
        raise UnregisteredOutputError(
            f"Output type {qualified!r} is not registered on this session. "
            f"Registered outputs: {names}. "
            "Add it to outputs= when building the session so the tool set stays fixed."
        )

    def prepare_query(self, query: Query[Any]) -> None:
        """Assign an id, reject unknown outputs, and warn on a broken continuation."""
        self.require_output(query.output)
        self._check_output_mode(query, diagnose=False)
        if not query.id:
            self._ids += 1
            query.id = f"q{self._ids}"
        warn_if_prefix_mismatch(
            query.history,
            self.rendered_documents,
            prefix_hash_value=self.prefix_hash,
        )

    def _check_output_mode(self, query: Query[Any], *, diagnose: bool = True) -> None:
        settings = effective_model_settings(self.model, self.model_settings, query.model_settings)
        if self.output_mode == "auto":
            actual = resolve_output_mode(self.model, settings)
            if actual != self.resolved_output_mode:
                raise OutputModeConflictError(
                    f"Query settings would change the frozen output mode from {self.resolved_output_mode!r} "
                    f"to {actual!r}. Build a separate session for these settings or choose an explicit output_mode.",
                    code="output_mode_conflict",
                )
        elif diagnose and self.output_mode == "tools" and self._forced_tools_available:
            if not forced_tools_available(self.model, settings):
                diagnose_unforced_tools(self.on_unforced_tools)

    def history_for(self, query: Query[Any]) -> list[ModelMessage]:
        """Prefix messages, or a copy of the continuation history."""
        if query.history is None:
            return prefix_messages(self.rendered_documents)
        return copy.deepcopy(query.history)

    def render_tail(self, query: Query[Any]) -> str:
        """Query text that follows the frozen prefix. Schema lives here in prompted mode."""
        name = self.require_output(query.output)
        parts = [query.tail.rstrip(), "", f"Use the `{name}` output and no other output type."]
        if self.resolved_output_mode == "prompted":
            schema = json.dumps(
                query.output.model_json_schema(),
                sort_keys=True,
                ensure_ascii=False,
            )
            parts.append("Respond with a single JSON object matching this schema:")
            parts.append(schema)
        return "\n".join(parts)

    async def ask(self, query: Query[T]) -> QueryResult[T]:
        """Run one query, with the session retry and concurrency policy."""
        self.prepare_query(query)
        result = await self._executor.run_query(self, query)
        return cast(QueryResult[T], result)

    async def fork(self, queries: Sequence[Query[Any]]) -> list[QueryResult[Any]]:
        """Run queries concurrently.

        When ``warmup`` is set, the first query of the first batch finishes
        before any other query starts, so the provider can store the prefix.
        Later batches run at full concurrency.
        """
        import asyncio

        pending = list(queries)
        for query in pending:
            self.prepare_query(query)
        if not pending:
            return []
        if self.warmup and not self._warmed and len(pending) > 1:
            first = await self._executor.run_query(self, pending[0])
            self._warmed = True
            rest = await asyncio.gather(*(self._executor.run_query(self, query) for query in pending[1:]))
            return [first, *rest]
        self._warmed = True
        return list(await asyncio.gather(*(self._executor.run_query(self, query) for query in pending)))

    async def run_once(self, query: Query[T]) -> QueryResult[T]:
        """One model call, with no library-level retry.

        Budget is checked here. Validation and transport errors come back as
        ``result.failure`` instead of being raised, except for caller errors
        such as an unregistered output type.
        """
        self.require_output(query.output)
        self._check_output_mode(query)
        if not query.id:
            self.prepare_query(query)
        notice = request_context_warning(self.policy, self.estimate_tokens(query))
        if notice is not None:
            emit_context_warning(notice, stacklevel=3)
        async with self._budget.token_turn():
            problem = await self._budget.try_reserve()
            if problem is not None:
                allowed = budget_action_failure(problem, self.policy)
                if allowed is not None:
                    return self._empty_result(query, allowed, attempts=0, context_warning=notice)
                await self._budget.force_reserve()
            started = time.perf_counter()
            messages: list[ModelMessage] = self.history_for(query)
            kwargs: dict[str, Any] = {}
            if self.resolved_output_mode == "prompted":
                kwargs["output_type"] = PromptedOutput(query.output, template=False)
            merged_settings = merge_model_settings(self.model_settings, query.model_settings)
            if merged_settings:
                # Copy so a later mutation of the caller's dict cannot change
                # this request, and so pydantic-ai's in-place edits stay local.
                kwargs["model_settings"] = dict(merged_settings)
            try:
                with capture_run_messages() as captured:
                    try:
                        result = await self._agent.run(
                            self.render_tail(query),
                            message_history=messages,
                            usage_limits=_REQUEST_LIMIT,
                            **kwargs,
                        )
                    except Exception as exc:
                        elapsed = (time.perf_counter() - started) * 1000
                        transcript = list(captured) or messages
                        # Validation failures still billed a response. Recover it
                        # from the transcript; a transport error has none.
                        billed = usage_from_messages(transcript)
                        await self._budget.add_usage(billed)
                        return self._empty_result(
                            query,
                            classify_exception(exc),
                            messages=transcript,
                            elapsed_ms=elapsed,
                            usage=billed,
                            context_warning=notice,
                        )
            except Exception as exc:
                # capture_run_messages itself should not fail; treat it as permanent.
                elapsed = (time.perf_counter() - started) * 1000
                return self._empty_result(
                    query,
                    classify_exception(exc),
                    elapsed_ms=elapsed,
                    context_warning=notice,
                )
            elapsed = (time.perf_counter() - started) * 1000
            await self._budget.add_usage(result.usage)
            hits = cache_hit_tokens(result.usage)
            transcript = list(result.all_messages())
            output = result.output
            if not isinstance(output, query.output):
                return QueryResult(
                    query_id=query.id or "",
                    output=None,
                    messages=transcript,
                    prefix_hash=self.prefix_hash,
                    resolved_output_mode=self.resolved_output_mode,
                    usage=result.usage,
                    cache_hit_tokens=hits,
                    attempts=1,
                    failure=wrong_output_failure(query.output.__name__, type(output).__name__),
                    round=query.round,
                    elapsed_ms=elapsed,
                    context_warning=notice,
                )
            return QueryResult(
                query_id=query.id or "",
                output=output,
                messages=transcript,
                prefix_hash=self.prefix_hash,
                resolved_output_mode=self.resolved_output_mode,
                usage=result.usage,
                cache_hit_tokens=hits,
                attempts=1,
                failure=None,
                round=query.round,
                elapsed_ms=elapsed,
                context_warning=notice,
            )

    def estimate_tokens(self, query: Query[Any]) -> int:
        """Estimated input tokens for one query, including the frozen prefix.

        See :func:`forkreduce.context.estimate_tokens`. This does not call the model,
        does not shorten the prompt, and does not decide whether the call is sent.
        """
        return estimate_tokens(
            request_text(
                instructions=self.instructions,
                rendered_documents=self.rendered_documents,
                tool_specs=self.tool_specs,
                tail=self.render_tail(query),
                history=query.history,
            )
        )

    def starts_with_prefix(self, messages: Sequence[ModelMessage]) -> bool:
        """Whether a transcript still begins with this session's documents."""
        return history_matches_prefix(messages, self.rendered_documents)

    async def reduce(
        self,
        items: Sequence[Any],
        *,
        output: type[Any],
        instructions: str = "",
        check: Any = None,
        max_batch: int = 200,
        coverage: Any = None,
        max_rounds: int = 3,
        id_of: Any = None,
        round_index: int = 0,
    ) -> Any:
        """Semantically merge ``items`` with a query on this same prefix.

        See :func:`forkreduce.reduce.reduce_items`. Imported lazily so importing
        a session does not import the reduce pipeline.
        """
        from forkreduce.reduce import reduce_items

        return await reduce_items(
            self,
            items,
            output=output,
            instructions=instructions,
            check=check,
            max_batch=max_batch,
            coverage=coverage,
            max_rounds=max_rounds,
            id_of=id_of,
            round_index=round_index,
        )

    def map(self, queries: Sequence[Query[Any]]) -> Any:
        """Start a :class:`forkreduce.plan.Plan` from these queries."""
        from forkreduce.plan import Plan

        return Plan(self, list(queries))

    def _empty_result(
        self,
        query: Query[Any],
        failure: Failure,
        *,
        messages: list[ModelMessage] | None = None,
        elapsed_ms: float = 0.0,
        usage: RunUsage | None = None,
        attempts: int = 1,
        context_warning: ContextWarning | None = None,
    ) -> QueryResult[Any]:
        recorded = usage if usage is not None else RunUsage()
        return QueryResult(
            query_id=query.id or "",
            output=None,
            messages=messages if messages is not None else self.history_for(query),
            prefix_hash=self.prefix_hash,
            resolved_output_mode=self.resolved_output_mode,
            usage=recorded,
            cache_hit_tokens=cache_hit_tokens(recorded),
            attempts=attempts,
            failure=failure,
            round=query.round,
            elapsed_ms=elapsed_ms,
            context_warning=context_warning,
        )


def build(
    model: Model,
    *,
    documents: DocumentsInput,
    instructions: str,
    outputs: Sequence[type[BaseModel]],
    output_mode: OutputMode = "auto",
    on_unforced_tools: UnforcedToolsPolicy = "warn",
    warmup: bool = True,
    policy: Policy | None = None,
    executor: Executor | None = None,
    model_settings: ModelSettings | None = None,
    activity_options: ActivityOptions | None = None,
    meta_keys: Sequence[str] | None = None,
    meta_renderer: MetaRenderer | None = None,
) -> Session:
    """Freeze a shared prefix and the output tool set.

    ``outputs`` is every structured type any query, repair, or reduce on this
    session will ask for. Adding one later is refused: a new tool schema would
    invalidate the provider prefix cache.

    ``output_mode="auto"`` (the default) freezes tools when the model profile
    and effective thinking settings permit forced output tools, otherwise
    prompted mode. Explicit ``"tools"`` registers every output on every call;
    ``"prompted"`` appends only the expected output's JSON schema to the tail.
    ``on_unforced_tools`` controls diagnostics for explicit tools when forcing
    is unavailable: ``"warn"`` (default), ``"error"``, or ``"ignore"``.

    ``model_settings`` is sent on every call. A :class:`~forkreduce.query.Query`
    may override individual keys. Sampling parameters and timeouts do not change
    the frozen prefix. ``meta_keys`` or ``meta_renderer`` do: rendered block
    meta becomes part of the prefix and its hash. The default renders no meta.

    ``activity_options`` sets Temporal-only activity timeouts and retry options.
    Durations are seconds. It does not change local execution or the prefix;
    leaving it unset preserves the five-minute activity timeout and retries
    derived from ``policy``.
    """
    if output_mode not in ("auto", "tools", "prompted"):
        raise ForkReduceError("output_mode must be 'auto', 'tools', or 'prompted'")
    if on_unforced_tools not in ("warn", "error", "ignore"):
        raise ForkReduceError("on_unforced_tools must be 'warn', 'error', or 'ignore'")
    settings = effective_model_settings(model, model_settings)
    resolved = resolve_output_mode(model, settings) if output_mode == "auto" else output_mode
    if output_mode == "tools" and not forced_tools_available(model, settings):
        diagnose_unforced_tools(on_unforced_tools)
    if not outputs:
        raise ForkReduceError("outputs must register at least one pydantic model")
    index = coerce_documents(documents, meta_keys=meta_keys, meta_renderer=meta_renderer)
    chosen = policy if policy is not None else Policy()
    output_types: dict[str, type[BaseModel]] = {}
    tool_specs: list[dict[str, Any]] = []
    for output in outputs:
        if not inspect.isclass(output) or not issubclass(output, BaseModel):
            raise ForkReduceError(f"output {output!r} must be a pydantic BaseModel subclass")
        name = output.__name__
        if name in output_types:
            raise ForkReduceError(f"duplicate output name {name!r}")
        description = inspect.getdoc(output) or f"Return a {name}."
        output_types[name] = output
        tool_specs.append(
            {
                "name": name,
                "description": description,
                "schema": output.model_json_schema(),
            }
        )
    hashed = prefix_hash(
        output_mode=resolved,
        instructions=instructions,
        rendered_documents=index.rendered,
        tools=tool_specs,
    )
    session = Session(
        model=model,
        instructions=instructions,
        documents=index,
        output_types=output_types,
        output_mode=output_mode,
        resolved_output_mode=resolved,
        on_unforced_tools=on_unforced_tools,
        policy=chosen,
        warmup=warmup,
        prefix_hash_value=hashed,
        tool_specs=tool_specs,
        executor=executor,
        model_settings=model_settings,
        activity_options=activity_options,
    )
    notice = prefix_context_warning(
        chosen,
        estimate_tokens(prefix_text(instructions, index.rendered, tool_specs)),
    )
    if notice is not None:
        session.context_warning = notice
        emit_context_warning(notice, stacklevel=3)
    return session


def _build_agent(
    model: Model,
    *,
    instructions: str,
    output_types: dict[str, type[BaseModel]],
    output_mode: ResolvedOutputMode,
    descriptions: dict[str, str],
) -> Agent[None, Any]:
    if output_mode == "tools":
        # max_retries=0 so a bad payload raises instead of spending a second
        # model call inside pydantic-ai. forkreduce owns the retry budget.
        output_spec: Any = [
            ToolOutput(cls, name=name, description=descriptions.get(name), max_retries=0)
            for name, cls in output_types.items()
        ]
    else:
        # template=False keeps the union schema out of the instruction prefix.
        # Each call passes its own PromptedOutput so the tail schema is the one
        # that gets validated.
        output_spec = PromptedOutput(list(output_types.values()), template=False)
    return Agent(
        model,
        output_type=output_spec,
        instructions=instructions,
        retries=0,
        name="forkreduce",
    )
