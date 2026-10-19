"""Sessions: a frozen prefix and the queries that fork off it."""

from __future__ import annotations

import copy
import inspect
import json
import time
from collections.abc import Sequence
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput, ToolOutput, capture_run_messages
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from forkreduce.documents import DocumentIndex, DocumentsInput, coerce_documents
from forkreduce.errors import ForkReduceError, UnregisteredOutputError
from forkreduce.executor import BudgetLedger, Executor, LocalExecutor, budget_action_failure
from forkreduce.failures import Failure, classify_exception, wrong_output_failure
from forkreduce.policy import Policy
from forkreduce.prefix import (
    history_matches_prefix,
    prefix_hash,
    prefix_messages,
    warn_if_prefix_mismatch,
)
from forkreduce.query import Query, QueryResult
from forkreduce.stats import cache_hit_tokens

T = TypeVar("T")
OutputMode = Literal["tools", "prompted"]

_REQUEST_LIMIT = UsageLimits(request_limit=1)


class Session:
    """A frozen prefix plus the registered output tools.

    Build one with :func:`forkreduce.build`. Queries share ``instructions``,
    the rendered documents, and (in ``tools`` mode) the whole output tool set.
    The per-query text is only the tail.
    """

    def __init__(
        self,
        *,
        model: Model,
        instructions: str,
        documents: DocumentIndex,
        output_types: dict[str, type[BaseModel]],
        output_mode: OutputMode,
        policy: Policy,
        warmup: bool,
        prefix_hash_value: str,
        tool_specs: list[dict[str, Any]],
        executor: Executor | None,
    ) -> None:
        self.model = model
        self.instructions = instructions
        self.documents = documents
        self.rendered_documents = documents.rendered
        self.output_types = dict(output_types)
        self.output_mode: OutputMode = output_mode
        self.policy = policy
        self.warmup = warmup
        self.prefix_hash = prefix_hash_value
        self.tool_specs = tool_specs
        self._executor: Executor = executor if executor is not None else LocalExecutor(policy)
        self._budget = BudgetLedger(policy)
        self._warmed = False
        self._ids = 0
        self._agent: Agent[None, Any] = _build_agent(
            model,
            instructions=instructions,
            output_types=self.output_types,
            output_mode=output_mode,
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
        if not query.id:
            self._ids += 1
            query.id = f"q{self._ids}"
        warn_if_prefix_mismatch(
            query.history,
            self.rendered_documents,
            prefix_hash_value=self.prefix_hash,
        )

    def history_for(self, query: Query[Any]) -> list[ModelMessage]:
        """Prefix messages, or a copy of the continuation history."""
        if query.history is None:
            return prefix_messages(self.rendered_documents)
        return copy.deepcopy(query.history)

    def render_tail(self, query: Query[Any]) -> str:
        """Query text that follows the frozen prefix. Schema lives here in prompted mode."""
        name = self.require_output(query.output)
        parts = [query.tail.rstrip(), "", f"Use the `{name}` output and no other output type."]
        if self.output_mode == "prompted":
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
        if not query.id:
            self.prepare_query(query)
        problem = await self._budget.try_reserve()
        if problem is not None:
            allowed = budget_action_failure(problem, self.policy)
            if allowed is not None:
                return self._empty_result(query, allowed)
            await self._budget.force_reserve()
        started = time.perf_counter()
        messages: list[ModelMessage] = self.history_for(query)
        kwargs: dict[str, Any] = {}
        if self.output_mode == "prompted":
            kwargs["output_type"] = PromptedOutput(query.output, template=False)
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
                    return self._empty_result(
                        query,
                        classify_exception(exc),
                        messages=transcript,
                        elapsed_ms=elapsed,
                    )
        except Exception as exc:
            # capture_run_messages itself should not fail; treat it as permanent.
            elapsed = (time.perf_counter() - started) * 1000
            return self._empty_result(query, classify_exception(exc), elapsed_ms=elapsed)
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
                usage=result.usage,
                cache_hit_tokens=hits,
                attempts=1,
                failure=wrong_output_failure(query.output.__name__, type(output).__name__),
                round=query.round,
                elapsed_ms=elapsed,
            )
        return QueryResult(
            query_id=query.id or "",
            output=output,
            messages=transcript,
            prefix_hash=self.prefix_hash,
            usage=result.usage,
            cache_hit_tokens=hits,
            attempts=1,
            failure=None,
            round=query.round,
            elapsed_ms=elapsed,
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
    ) -> QueryResult[Any]:
        return QueryResult(
            query_id=query.id or "",
            output=None,
            messages=messages if messages is not None else self.history_for(query),
            prefix_hash=self.prefix_hash,
            usage=RunUsage(),
            cache_hit_tokens=0,
            attempts=1,
            failure=failure,
            round=query.round,
            elapsed_ms=elapsed_ms,
        )


def build(
    model: Model,
    *,
    documents: DocumentsInput,
    instructions: str,
    outputs: Sequence[type[BaseModel]],
    output_mode: OutputMode = "tools",
    warmup: bool = True,
    policy: Policy | None = None,
    executor: Executor | None = None,
) -> Session:
    """Freeze a shared prefix and the output tool set.

    ``outputs`` is every structured type any query, repair, or reduce on this
    session will ask for. Adding one later is refused: a new tool schema would
    invalidate the provider prefix cache.

    ``output_mode="tools"`` (the default) registers every output as a tool on
    every call. ``output_mode="prompted"`` keeps tools out of the prefix and
    appends only the expected output's JSON schema to the query tail.
    """
    if output_mode not in {"tools", "prompted"}:
        raise ForkReduceError("output_mode must be 'tools' or 'prompted'")
    if not outputs:
        raise ForkReduceError("outputs must register at least one pydantic model")
    index = coerce_documents(documents)
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
        output_mode=output_mode,
        instructions=instructions,
        rendered_documents=index.rendered,
        tools=tool_specs,
    )
    return Session(
        model=model,
        instructions=instructions,
        documents=index,
        output_types=output_types,
        output_mode=output_mode,
        policy=chosen,
        warmup=warmup,
        prefix_hash_value=hashed,
        tool_specs=tool_specs,
        executor=executor,
    )


def _build_agent(
    model: Model,
    *,
    instructions: str,
    output_types: dict[str, type[BaseModel]],
    output_mode: OutputMode,
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
