"""Semantic reduce on the same frozen prefix."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from forkreduce.coverage import (
    AcceptUncovered,
    Provenance,
    ProvenanceCheck,
    ReduceReport,
    iter_traced,
    rewrite_source_ids,
)
from forkreduce.errors import ForkReduceError
from forkreduce.failures import Failure, FailureKind
from forkreduce.policy import Accept, Drop, Fail, RequireReason, Retry, action_name
from forkreduce.query import Query, QueryResult
from forkreduce.schemas import DiscardExplanation, DiscardExplanations
from forkreduce.session import Session
from forkreduce.stats import add_usage

_IdOf = Callable[[Any], str | None]


@dataclass
class ReduceItem:
    """One identified input to a reduce."""

    id: str
    value: Any
    round: int = 0


@dataclass
class ReduceOutcome:
    """Result of :func:`reduce_items`, including the coverage report and every model call."""

    output: Any
    report: ReduceReport
    result: QueryResult[Any] | None
    calls: list[QueryResult[Any]] = field(default_factory=list)
    failure: Failure | None = None
    fatal: bool = False
    reasons: list[DiscardExplanation] = field(default_factory=list)
    action: str = ""


@dataclass
class _Level:
    output: Any
    calls: list[QueryResult[Any]]
    failure: Failure | None = None


async def reduce_items(
    session: Session,
    items: Sequence[Any],
    *,
    output: type[Any],
    instructions: str = "",
    check: ProvenanceCheck | None = None,
    max_batch: int = 200,
    coverage: Any = None,
    max_rounds: int = 3,
    id_of: _IdOf | None = None,
    round_index: int = 0,
) -> ReduceOutcome:
    """Merge ``items`` by asking the model, on this session's prefix.

    The reduce call is an ordinary query, so it hits the same prefix cache.
    Outputs must carry ``source_ids``. When there are more than ``max_batch``
    items the inputs are reduced in a tree; batch ids are expanded back to the
    original ids before the coverage policy runs.

    The default coverage policy accepts the merge. Uncovered ids stay on
    :class:`ReduceReport` either way.
    """
    if max_batch < 1:
        raise ForkReduceError("max_batch must be >= 1")
    if max_rounds < 1:
        raise ForkReduceError("max_rounds must be >= 1")
    identified = identify_items(items, id_of)
    if not identified:
        raise ForkReduceError("reduce() requires at least one item")
    session.require_output(output)
    checker: ProvenanceCheck = check if check is not None else Provenance()
    policy = coverage if coverage is not None else AcceptUncovered()
    original_ids = [item.id for item in identified]
    alias: dict[str, list[str]] = {}
    counter = _Counter()
    current = list(identified)
    hint = ""
    merged: Any = None
    calls: list[QueryResult[Any]] = []
    report = ReduceReport(covered=[], uncovered=list(original_ids), invalid_refs=[])

    for attempt in range(max_rounds):
        alias.clear()
        level = await _reduce_tree(
            session,
            current,
            output=output,
            instructions=instructions,
            hint=hint,
            checker=checker,
            max_batch=max_batch,
            alias=alias,
            counter=counter,
            round_index=round_index,
        )
        calls = level.calls
        if level.output is None:
            failure = level.failure or Failure(
                kind=FailureKind.PERMANENT,
                message="reduce produced no output",
                retryable=False,
                code="reduce_failed",
            )
            return ReduceOutcome(
                output=None,
                report=ReduceReport(covered=[], uncovered=list(original_ids), invalid_refs=[]),
                result=_combine(calls),
                calls=calls,
                failure=failure,
                fatal=True,
                action="Fail",
            )
        merged = rewrite_source_ids(level.output, alias)
        if round_index > 0:
            merged = annotate_round(merged, round_index)
        report = checker.evaluate(merged, original_ids)
        action = policy.decide(report)
        combined = _combine(calls)
        if isinstance(action, (Accept, Drop)):
            return ReduceOutcome(
                output=merged,
                report=report,
                result=combined,
                calls=calls,
                action=action_name(action),
            )
        if isinstance(action, Fail):
            return ReduceOutcome(
                output=merged,
                report=report,
                result=combined,
                calls=calls,
                failure=Failure(
                    kind=FailureKind.PERMANENT,
                    message=action.reason,
                    retryable=False,
                    code="coverage_failed",
                ),
                fatal=True,
                action="Fail",
            )
        if isinstance(action, RequireReason):
            reasons, reason_calls, reason_failure = await _ask_reasons(
                session,
                combined,
                action.items,
                counter,
                round_index,
            )
            calls = [*calls, *reason_calls]
            report.reasons = list(reasons)
            return ReduceOutcome(
                output=merged,
                report=report,
                result=_combine(calls),
                calls=calls,
                failure=reason_failure,
                fatal=False,
                reasons=reasons,
                action="RequireReason",
            )
        if isinstance(action, Retry):
            if attempt >= max_rounds - 1:
                break
            hint = action.hint
            if action.items:
                wanted = set(action.items)
                filtered = [item for item in identified if item.id in wanted]
                current = filtered or list(identified)
            else:
                current = list(identified)
            continue
        return ReduceOutcome(
            output=merged,
            report=report,
            result=combined,
            calls=calls,
            failure=Failure(
                kind=FailureKind.PERMANENT,
                message=f"coverage policy returned unsupported action {action_name(action)}",
                retryable=False,
                code="bad_action",
            ),
            fatal=True,
            action=action_name(action),
        )

    return ReduceOutcome(
        output=merged,
        report=report,
        result=_combine(calls),
        calls=calls,
        failure=Failure(
            kind=FailureKind.PERMANENT,
            message=f"coverage retry limit reached ({max_rounds})",
            retryable=False,
            code="coverage_retries",
        ),
        fatal=True,
        action="Fail",
    )


def identify_items(items: Sequence[Any], id_of: _IdOf | None) -> list[ReduceItem]:
    """Assign a stable id to each reduce input."""
    identified: list[ReduceItem] = []
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        if isinstance(item, ReduceItem):
            item_id, value, item_round = item.id, item.value, item.round
        elif isinstance(item, QueryResult):
            if item.output is None:
                raise ForkReduceError(f"cannot reduce query {item.query_id!r}: it has no output")
            item_id, value, item_round = item.query_id, item.output, item.round
        else:
            custom = id_of(item) if id_of is not None else None
            if custom:
                item_id = str(custom)
            elif isinstance(item, BaseModel) and "id" in type(item).model_fields:
                raw_id = getattr(item, "id")  # noqa: B009 - field is optional on BaseModel
                item_id = str(raw_id) if isinstance(raw_id, str) and raw_id else f"item-{index}"
            else:
                item_id = f"item-{index}"
            value, item_round = item, 0
        if item_id in seen:
            raise ForkReduceError(f"duplicate reduce item id {item_id!r}")
        seen.add(item_id)
        identified.append(ReduceItem(id=item_id, value=value, round=item_round))
    return identified


def flatten_merged(merged: Any, *, round_index: int) -> list[ReduceItem]:
    """Turn a previous merge into inputs for the next round.

    Traced entries keep their own ids when they cite exactly one source, so a
    later reduce can still point at them. Otherwise they get a ``prev-`` id.
    """
    if merged is None:
        return []
    nodes = iter_traced(merged)
    if not nodes:
        return [ReduceItem(id=f"prev-r{round_index}", value=merged, round=round_index)]
    items: list[ReduceItem] = []
    seen: set[str] = set()
    for index, node in enumerate(nodes):
        item_id = node.source_ids[0] if len(node.source_ids) == 1 else f"prev-r{round_index}-{index}"
        if item_id in seen:
            item_id = f"{item_id}-{index}"
        seen.add(item_id)
        items.append(ReduceItem(id=item_id, value=node.value, round=round_index))
    return items


def annotate_round(value: Any, round_index: int) -> Any:
    """Set ``round`` on models that declare that field, including nested items."""
    if isinstance(value, list):
        return [annotate_round(item, round_index) for item in value]
    if not isinstance(value, BaseModel):
        return value
    updates: dict[str, Any] = {}
    if "round" in type(value).model_fields:
        updates["round"] = round_index
    for name in type(value).model_fields:
        if name == "round":
            continue
        child = getattr(value, name)
        if isinstance(child, BaseModel):
            rewritten = annotate_round(child, round_index)
            if rewritten is not child:
                updates[name] = rewritten
        elif isinstance(child, list):
            rewritten_list = [annotate_round(item, round_index) for item in child]
            if any(new is not old for new, old in zip(rewritten_list, child, strict=False)):
                updates[name] = rewritten_list
    if not updates:
        return value
    return value.model_copy(update=updates)


async def _reduce_tree(
    session: Session,
    items: list[ReduceItem],
    *,
    output: type[Any],
    instructions: str,
    hint: str,
    checker: ProvenanceCheck,
    max_batch: int,
    alias: dict[str, list[str]],
    counter: _Counter,
    round_index: int,
) -> _Level:
    if len(items) > max_batch:
        calls: list[QueryResult[Any]] = []
        flat: list[ReduceItem] = []
        for part in _chunks(items, max_batch):
            level = await _reduce_tree(
                session,
                part,
                output=output,
                instructions=instructions,
                hint=hint,
                checker=checker,
                max_batch=max_batch,
                alias=alias,
                counter=counter,
                round_index=round_index,
            )
            calls.extend(level.calls)
            if level.output is None:
                return _Level(output=None, calls=calls, failure=level.failure)
            nodes = iter_traced(level.output)
            if not nodes:
                new_id = counter.next("m")
                alias[new_id] = [item.id for item in part]
                flat.append(ReduceItem(id=new_id, value=level.output, round=round_index))
            else:
                for node in nodes:
                    new_id = counter.next("m")
                    alias[new_id] = list(node.source_ids)
                    flat.append(ReduceItem(id=new_id, value=node.value, round=round_index))
        parent = await _reduce_tree(
            session,
            flat,
            output=output,
            instructions=instructions,
            hint=hint,
            checker=checker,
            max_batch=max_batch,
            alias=alias,
            counter=counter,
            round_index=round_index,
        )
        return _Level(output=parent.output, calls=[*calls, *parent.calls], failure=parent.failure)

    return await _reduce_batch(
        session,
        items,
        output=output,
        instructions=instructions,
        hint=hint,
        checker=checker,
        counter=counter,
        round_index=round_index,
    )


async def _reduce_batch(
    session: Session,
    items: list[ReduceItem],
    *,
    output: type[Any],
    instructions: str,
    hint: str,
    checker: ProvenanceCheck,
    counter: _Counter,
    round_index: int,
) -> _Level:
    calls: list[QueryResult[Any]] = []
    local_hint = hint
    attempts = session.policy.retries + 1
    last_output: Any = None
    for attempt in range(attempts):
        query: Query[Any] = Query(
            tail=_render_reduce_tail(items, instructions, local_hint),
            output=output,
            id=f"reduce-r{round_index}-{counter.next('c')}",
            round=round_index,
        )
        result = await session.ask(query)
        calls.append(result)
        if result.failure is not None or result.output is None:
            return _Level(output=None, calls=calls, failure=result.failure)
        last_output = result.output
        local = checker.evaluate(result.output, [item.id for item in items])
        if not local.invalid_refs and not local.untraced:
            return _Level(output=result.output, calls=calls, failure=None)
        local_hint = (
            "The previous output had invalid or missing source_ids. "
            "Cite only ids from the input list, and put source_ids on every entry. " + hint
        )
        if attempt >= attempts - 1:
            return _Level(output=last_output, calls=calls, failure=None)
    return _Level(output=last_output, calls=calls, failure=None)


async def _ask_reasons(
    session: Session,
    merged_result: QueryResult[Any] | None,
    item_ids: tuple[str, ...],
    counter: _Counter,
    round_index: int,
) -> tuple[list[DiscardExplanation], list[QueryResult[Any]], Failure | None]:
    if merged_result is None:
        return (
            [],
            [],
            Failure(
                kind=FailureKind.PERMANENT,
                message="cannot ask for discard reasons without a reduce branch",
                retryable=False,
                code="reasons_unavailable",
            ),
        )
    query: Query[DiscardExplanations] = Query(
        tail=_render_reason_tail(item_ids),
        output=DiscardExplanations,
        id=f"reasons-r{round_index}-{counter.next('c')}",
        history=list(merged_result.messages),
        round=round_index,
    )
    try:
        result = await session.ask(query)
    except ForkReduceError as exc:
        return (
            [],
            [],
            Failure(
                kind=FailureKind.PERMANENT,
                message=str(exc),
                retryable=False,
                code="unregistered_output",
            ),
        )
    if result.failure is not None or not isinstance(result.output, DiscardExplanations):
        failure = result.failure or Failure(
            kind=FailureKind.VALIDATION,
            message="discard-reason call did not return DiscardExplanations",
            retryable=False,
            code="reasons_failed",
        )
        return [], [result], failure
    return list(result.output.explanations), [result], None


def _render_reduce_tail(items: Sequence[ReduceItem], instructions: str, hint: str) -> str:
    rendered = "\n\n".join(_render_item(item) for item in items)
    lines = [
        "Merge the input items into the requested output.",
        "Every output entry must include source_ids copied from the input ids below.",
        "Do not invent source ids.",
    ]
    if instructions.strip():
        lines.extend(["", instructions.strip()])
    if hint.strip():
        lines.extend(["", f"Hint: {hint.strip()}"])
    lines.extend(["", "INPUT_ITEMS", rendered])
    return "\n".join(lines)


def _render_reason_tail(item_ids: Sequence[str]) -> str:
    listed = "\n".join(f"- {item_id}" for item_id in item_ids)
    return (
        "These input ids were not covered by your merge.\n"
        "Explain each one. Do not change the merge itself.\n"
        "Use the DiscardExplanations output.\n\n"
        f"{listed}"
    )


def _render_item(item: ReduceItem) -> str:
    value = item.value
    if isinstance(value, BaseModel):
        body = value.model_dump_json()
    else:
        body = json.dumps(value, default=str, ensure_ascii=False)
    return f"id: {item.id}\n{body}"


def _combine(calls: list[QueryResult[Any]]) -> QueryResult[Any] | None:
    if not calls:
        return None
    usage = RunUsage()
    cache = 0
    elapsed = 0.0
    attempts = 0
    for call in calls:
        usage = add_usage(usage, call.usage)
        cache += call.cache_hit_tokens
        elapsed += call.elapsed_ms
        attempts += call.attempts
    return replace(
        calls[-1],
        usage=usage,
        cache_hit_tokens=cache,
        elapsed_ms=elapsed,
        attempts=attempts,
    )


def _chunks(items: list[ReduceItem], size: int) -> list[list[ReduceItem]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


class _Counter:
    def __init__(self) -> None:
        self._value = 0

    def next(self, prefix: str) -> str:
        self._value += 1
        return f"{prefix}{self._value}"
