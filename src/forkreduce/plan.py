"""Chained map / check / filter / reduce / followup plan."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai.usage import RunUsage

from forkreduce.checks import Check, CheckReport, Repair
from forkreduce.coverage import ReduceReport
from forkreduce.failures import Failure, FailureKind
from forkreduce.followup import FollowupFn, RoundState
from forkreduce.query import Query, QueryResult
from forkreduce.reduce import (
    ReduceItem,
    annotate_round,
    flatten_merged,
    reduce_items,
)
from forkreduce.session import Session
from forkreduce.stats import UsageSummary, add_usage, summarize

FilterFn = Callable[[Any], bool]


@dataclass
class RoundRecord:
    """One map/check/reduce round."""

    round: int
    results: list[QueryResult[Any]]
    outputs: list[Any]
    merged: Any
    report: ReduceReport | None
    failures: list[Failure]
    check_reports: list[CheckReport] = field(default_factory=list)


@dataclass
class RunResult:
    """The full plan: merged output, per-round detail, usage, and every failure."""

    merged: Any
    rounds: list[RoundRecord]
    results: list[QueryResult[Any]]
    usage: UsageSummary
    failures: list[Failure]
    reports: list[Any]

    @property
    def cache_hit_rate(self) -> float:
        return self.usage.cache_hit_rate


class Plan:
    """A lazy pipeline. Nothing runs until :meth:`run`."""

    def __init__(self, session: Session, queries: list[Query[Any]]) -> None:
        self._session = session
        self._queries = queries
        self._check: Check | None = None
        self._repair: Repair | None = None
        self._failure_mode = session.policy.repair_failure
        self._filters: list[FilterFn] = []
        self._reduce_output: type[Any] | None = None
        self._reduce_instructions = ""
        self._reduce_check: Any = None
        self._reduce_max_batch = 200
        self._reduce_coverage: Any = None
        self._reduce_max_rounds = 3
        self._followup: FollowupFn | None = None
        self._max_rounds = 1

    def check(
        self,
        check: Check,
        *,
        repair: Repair | None = None,
        on_failure: str | None = None,
    ) -> Plan:
        """Validate each output. ``repair`` may continue the branch to fix it.

        ``on_failure`` overrides :attr:`Policy.repair_failure` for items that
        are still failing when repair stops: ``needs_review``, ``drop``, or
        ``fail``.
        """
        if on_failure is not None and on_failure not in {"needs_review", "drop", "fail"}:
            raise ValueError("on_failure must be 'needs_review', 'drop', or 'fail'")
        self._check = check
        self._repair = repair
        if on_failure is not None:
            self._failure_mode = on_failure
        return self

    def filter(self, fn: FilterFn) -> Plan:
        """Keep outputs for which ``fn`` returns true. Others stay on the run, not in reduce."""
        self._filters.append(fn)
        return self

    def reduce(
        self,
        *,
        output: type[Any],
        instructions: str = "",
        check: Any = None,
        max_batch: int = 200,
        coverage: Any = None,
        max_rounds: int = 3,
    ) -> Plan:
        """Semantically merge the kept outputs. ``output`` must be registered."""
        self._reduce_output = output
        self._reduce_instructions = instructions
        self._reduce_check = check
        self._reduce_max_batch = max_batch
        self._reduce_coverage = coverage
        self._reduce_max_rounds = max_rounds
        return self

    def followup(self, fn: FollowupFn, *, max_rounds: int = 2) -> Plan:
        """Call ``fn`` between rounds. ``max_rounds`` counts map rounds, including the first."""
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self._followup = fn
        self._max_rounds = max_rounds
        return self

    async def run(self) -> RunResult:
        """Execute the pipeline."""
        rounds: list[RoundRecord] = []
        all_results: list[QueryResult[Any]] = []
        all_failures: list[Failure] = []
        reports: list[Any] = []
        merged: Any = None
        carried: list[ReduceItem] = []
        queries = list(self._queries)
        fatal = False

        for round_index in range(1, self._max_rounds + 1):
            if not queries:
                break
            for query in queries:
                query.round = round_index
            results = await self._session.fork(queries)
            checked, check_reports, round_failures, stop = await self._apply_checks(results)
            all_failures.extend(round_failures)
            reports.extend(check_reports)
            kept: list[QueryResult[Any]] = []
            new_items: list[ReduceItem] = []
            for result in checked:
                if round_index > 0 and result.output is not None:
                    result.output = annotate_round(result.output, round_index)
                if result.dropped or result.output is None:
                    continue
                if result.failure is not None and not result.needs_review:
                    continue
                if not self._keep(result.output):
                    continue
                kept.append(result)
                new_items.append(ReduceItem(id=result.query_id, value=result.output, round=round_index))
            report: ReduceReport | None = None
            reduce_calls: list[QueryResult[Any]] = []
            if self._reduce_output is not None and (new_items or carried) and not stop:
                outcome = await reduce_items(
                    self._session,
                    [*carried, *new_items],
                    output=self._reduce_output,
                    instructions=self._reduce_instructions,
                    check=self._reduce_check,
                    max_batch=self._reduce_max_batch,
                    coverage=self._reduce_coverage,
                    max_rounds=self._reduce_max_rounds,
                    round_index=round_index,
                )
                merged = outcome.output
                report = outcome.report
                reports.append(report)
                reduce_calls = list(outcome.calls)
                if outcome.failure is not None:
                    all_failures.append(outcome.failure)
                if outcome.fatal:
                    fatal = True
                carried = flatten_merged(merged, round_index=round_index)
            round_results = [*checked, *reduce_calls]
            all_results.extend(round_results)
            rounds.append(
                RoundRecord(
                    round=round_index,
                    results=round_results,
                    outputs=[item.value for item in new_items],
                    merged=merged,
                    report=report,
                    failures=list(round_failures),
                    check_reports=check_reports,
                )
            )
            if fatal or stop or self._followup is None or round_index >= self._max_rounds:
                break
            state = RoundState(
                round=round_index,
                results=checked,
                outputs=[item.value for item in new_items],
                merged=merged,
                report=report,
            )
            nxt = self._followup(state, self._session)
            if inspect.isawaitable(nxt):
                nxt = await nxt
            all_results.extend(state.side_results)
            all_failures.extend(state.failures)
            # Side results were not part of this round record yet.
            if rounds:
                rounds[-1].results.extend(state.side_results)
                rounds[-1].failures.extend(state.failures)
            queries = list(nxt)
            if not queries:
                break

        return RunResult(
            merged=merged,
            rounds=rounds,
            results=all_results,
            usage=summarize(all_results),
            failures=all_failures,
            reports=reports,
        )

    async def _apply_checks(
        self,
        results: list[QueryResult[Any]],
    ) -> tuple[list[QueryResult[Any]], list[CheckReport], list[Failure], bool]:
        if self._check is None:
            return results, [], [result.failure for result in results if result.failure], False
        updated: list[QueryResult[Any]] = []
        reports: list[CheckReport] = []
        failures: list[Failure] = []
        stop = False
        for result in results:
            if result.failure is not None or result.output is None:
                if result.failure is not None:
                    failures.append(result.failure)
                updated.append(result)
                continue
            parts = [result]
            output = result.output
            messages = result.messages
            problems = self._check(output, documents=self._session.documents)
            repair_rounds = 0
            while problems and self._repair is not None and repair_rounds < self._repair.max_rounds:
                repair_rounds += 1
                carrier = replace(result, output=output, messages=messages, query_id=result.query_id)
                repaired = await self._session.ask(
                    self._repair.build_query(carrier, problems, round_index=repair_rounds)
                )
                parts.append(repaired)
                if repaired.failure is not None or repaired.output is None:
                    if repaired.failure is not None:
                        failures.append(repaired.failure)
                    break
                try:
                    output = self._repair.apply(output, repaired.output, problems)
                except Exception as exc:
                    failures.append(
                        Failure(
                            kind=FailureKind.PERMANENT,
                            message=f"repair apply failed: {exc}",
                            retryable=False,
                            code="repair_apply",
                        )
                    )
                    break
                messages = repaired.messages
                problems = self._check(output, documents=self._session.documents)
            usage, cache, attempts, elapsed = _fold_usage(parts)
            current = replace(
                result,
                output=output,
                messages=messages,
                usage=usage,
                cache_hit_tokens=cache,
                attempts=attempts,
                elapsed_ms=elapsed,
                failure=None,
                check_failures=list(problems),
            )
            if not problems:
                action = "repaired" if repair_rounds else "ok"
            else:
                action = self._failure_mode
                if self._failure_mode == "needs_review":
                    current.needs_review = True
                elif self._failure_mode == "drop":
                    current.dropped = True
                else:
                    failure = Failure(
                        kind=FailureKind.PERMANENT,
                        message="; ".join(item.message for item in problems),
                        retryable=False,
                        code="check_failed",
                    )
                    current.failure = failure
                    failures.append(failure)
                    stop = True
            reports.append(
                CheckReport(
                    query_id=result.query_id,
                    round=result.round,
                    failures=list(problems),
                    repair_rounds=repair_rounds,
                    action=action,
                )
            )
            updated.append(current)
        return updated, reports, failures, stop

    def _keep(self, output: Any) -> bool:
        return all(fn(output) for fn in self._filters)


def _fold_usage(parts: Sequence[QueryResult[Any]]) -> tuple[RunUsage, int, int, float]:
    usage = RunUsage()
    cache = 0
    attempts = 0
    elapsed = 0.0
    for part in parts:
        usage = add_usage(usage, part.usage)
        cache += part.cache_hit_tokens
        attempts += part.attempts
        elapsed += part.elapsed_ms
    return usage, cache, attempts, elapsed
