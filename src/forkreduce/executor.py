"""Single-attempt execution and the default local executor.

:class:`LocalExecutor` is what :meth:`Session.ask` uses: a semaphore, bounded
retries, and backoff for transport errors. One attempt lives on
:meth:`Session.run_once` so a Temporal activity can do the same call and let
Temporal own the retry.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from pydantic_ai.usage import RunUsage

from forkreduce.failures import Failure, budget_failure
from forkreduce.policy import Accept, BudgetReport, Fail, Policy
from forkreduce.query import Query, QueryResult
from forkreduce.stats import add_usage

if TYPE_CHECKING:
    from forkreduce.session import Session


class Executor(Protocol):
    """Runs one query to a :class:`QueryResult`, including whatever retries it owns."""

    async def run_query(self, session: Session, query: Query[Any]) -> QueryResult[Any]:
        """Execute ``query`` against ``session``."""
        ...


@dataclass
class BudgetLedger:
    policy: Policy
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: asyncio.Lock | None = None

    def _lock_for(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def try_reserve(self) -> BudgetReport | None:
        budget = self.policy.budget
        if budget is None:
            return None
        async with self._lock_for():
            problem = self._problem()
            if problem is not None:
                return problem
            self.requests += 1
            return None

    async def force_reserve(self) -> None:
        async with self._lock_for():
            self.requests += 1

    async def add_usage(self, usage: RunUsage) -> None:
        async with self._lock_for():
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens

    def _problem(self) -> BudgetReport | None:
        budget = self.policy.budget
        if budget is None:
            return None
        message: str | None = None
        if budget.max_requests is not None and self.requests >= budget.max_requests:
            message = f"request budget exhausted ({self.requests}/{budget.max_requests})"
        elif budget.max_input_tokens is not None and self.input_tokens >= budget.max_input_tokens:
            message = f"input token budget exhausted ({self.input_tokens}/{budget.max_input_tokens})"
        elif budget.max_output_tokens is not None and self.output_tokens >= budget.max_output_tokens:
            message = f"output token budget exhausted ({self.output_tokens}/{budget.max_output_tokens})"
        elif budget.max_total_tokens is not None and (
            self.input_tokens + self.output_tokens >= budget.max_total_tokens
        ):
            total = self.input_tokens + self.output_tokens
            message = f"total token budget exhausted ({total}/{budget.max_total_tokens})"
        if message is None:
            return None
        return BudgetReport(
            kind="budget",
            message=message,
            requests=self.requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


class LocalExecutor:
    """asyncio executor with a concurrency semaphore and bounded retries.

    Validation failures retry immediately. Transport and rate-limit failures
    sleep with exponential backoff. Permanent failures and budget exhaustion
    are returned on the result and are not retried.
    """

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self._semaphore = asyncio.Semaphore(policy.concurrency)

    async def run_query(self, session: Session, query: Query[Any]) -> QueryResult[Any]:
        async with self._semaphore:
            return await self._run_with_retries(session, query)

    async def _run_with_retries(self, session: Session, query: Query[Any]) -> QueryResult[Any]:
        usage = RunUsage()
        cache = 0
        elapsed = 0.0
        last: QueryResult[object] | None = None
        attempts = 0
        for attempt in range(self.policy.retries + 1):
            attempts = attempt + 1
            result = await session.run_once(query)
            usage = add_usage(usage, result.usage)
            cache += result.cache_hit_tokens
            elapsed += result.elapsed_ms
            last = result
            if result.failure is None:
                return self._folded(result, usage, cache, elapsed, attempts)
            if not result.failure.retryable or attempt >= self.policy.retries:
                return self._folded(result, usage, cache, elapsed, attempts)
            if result.failure.kind.value == "transport":
                await self._sleep(attempt + 1)
        assert last is not None
        return self._folded(last, usage, cache, elapsed, attempts)

    def _folded(
        self,
        result: QueryResult[object],
        usage: RunUsage,
        cache: int,
        elapsed: float,
        attempts: int,
    ) -> QueryResult[object]:
        failure = result.failure.with_attempts(attempts) if result.failure is not None else None
        return replace(
            result,
            usage=usage,
            cache_hit_tokens=cache,
            elapsed_ms=elapsed,
            attempts=attempts,
            failure=failure,
        )

    async def _sleep(self, attempt: int) -> None:
        delay = min(self.policy.retry_backoff * (2 ** (attempt - 1)), self.policy.max_backoff)
        if delay > 0:
            await asyncio.sleep(delay)


def budget_action_failure(report: BudgetReport, policy: Policy) -> Failure | None:
    """Apply the budget policy. ``None`` means the call may proceed anyway."""
    action: Accept | Fail
    if policy.budget_policy is None:
        action = Fail(report.message)
    else:
        decided = policy.budget_policy(report)
        if isinstance(decided, Accept):
            return None
        action = decided if isinstance(decided, Fail) else Fail(report.message)
    return budget_failure(action.reason)
