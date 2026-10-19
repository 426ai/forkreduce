"""Failure classification, retries, concurrency, warmup, budget, and usage."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError
from tests.support import Scripted, tool_result, usage

from forkreduce import Query, build
from forkreduce.failures import FailureKind, classify_exception
from forkreduce.policy import Accept, Budget, BudgetReport, Fail, Policy


class Answer(BaseModel):
    """A short answer."""

    text: str


def _ok(info, text="ok"):  # type: ignore[no-untyped-def]
    return tool_result(info, "Answer", {"text": text}, token_usage=usage(cache_read_tokens=80))


def test_classify_http_and_validation() -> None:
    retryable = classify_exception(ModelHTTPError(status_code=429, model_name="m", body="slow"))
    permanent = classify_exception(ModelHTTPError(status_code=401, model_name="m", body="no"))
    assert retryable.kind is FailureKind.TRANSPORT and retryable.retryable
    assert permanent.kind is FailureKind.PERMANENT and not permanent.retryable


@pytest.mark.asyncio
async def test_wrong_output_type_is_retried() -> None:
    def handler(_messages, info, n):  # type: ignore[no-untyped-def]
        if n == 1:
            return tool_result(info, "Other", {"body": "nope"})
        return _ok(info, "yes")

    class Other(BaseModel):
        """Something else."""

        body: str

    session = build(
        Scripted(handler).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer, Other],
        warmup=False,
        policy=Policy(retries=1, retry_backoff=0),
    )
    result = await session.ask(Query(tail="go", output=Answer, id="q"))
    assert result.failure is None
    assert result.output is not None and result.output.text == "yes"
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_validation_error_then_success() -> None:
    def handler(_messages, info, n):  # type: ignore[no-untyped-def]
        if n == 1:
            return tool_result(info, "Answer", {})
        return _ok(info)

    session = build(
        Scripted(handler).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=1, retry_backoff=0),
    )
    result = await session.ask(Query(tail="go", output=Answer, id="q"))
    assert result.output is not None and result.output.text == "ok"
    assert result.attempts == 2
    assert result.failure is None


@pytest.mark.asyncio
async def test_transport_retries_and_permanent_does_not() -> None:
    def flaky(_messages, info, n):  # type: ignore[no-untyped-def]
        if n == 1:
            raise ModelHTTPError(status_code=429, model_name="m", body="slow")
        return _ok(info)

    session = build(
        Scripted(flaky).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=1, retry_backoff=0),
    )
    result = await session.ask(Query(tail="go", output=Answer, id="q"))
    assert result.output is not None
    assert result.attempts == 2

    def denied(_messages, _info, _n):  # type: ignore[no-untyped-def]
        raise ModelHTTPError(status_code=401, model_name="m", body="no")

    session = build(
        Scripted(denied).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=3, retry_backoff=0),
    )
    failed = await session.ask(Query(tail="go", output=Answer, id="q"))
    assert failed.output is None
    assert failed.failure is not None
    assert failed.failure.kind is FailureKind.PERMANENT
    assert failed.attempts == 1


@pytest.mark.asyncio
async def test_concurrency_limit_and_warmup_order() -> None:
    current = 0
    peak = 0
    lock = asyncio.Lock()

    async def slow(_messages, info, _n):  # type: ignore[no-untyped-def]
        nonlocal current, peak
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return _ok(info)

    session = build(
        Scripted(slow).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(concurrency=2, retries=0, retry_backoff=0),
    )
    results = await session.fork([Query(tail=f"q{i}", output=Answer, id=f"q{i}") for i in range(6)])
    assert peak <= 2
    assert peak >= 2
    assert all(result.failure is None for result in results)

    order: list[tuple[str, str]] = []

    async def ordered(messages, info, _n):  # type: ignore[no-untyped-def]
        from tests.support import user_text

        text = user_text(messages)
        label = next(name for name in ("first", "second", "third") if name in text)
        order.append(("start", label))
        await asyncio.sleep(0.05)
        order.append(("end", label))
        return _ok(info)

    session = build(
        Scripted(ordered).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=True,
        policy=Policy(concurrency=4, retries=0, retry_backoff=0),
    )
    await session.fork(
        [
            Query(tail="first", output=Answer, id="a"),
            Query(tail="second", output=Answer, id="b"),
            Query(tail="third", output=Answer, id="c"),
        ]
    )
    end_first = order.index(("end", "first"))
    assert order.index(("start", "second")) > end_first
    assert order.index(("start", "third")) > end_first


@pytest.mark.asyncio
async def test_budget_stops_later_calls_and_can_be_overridden() -> None:
    calls = 0

    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return _ok(info)

    session = build(
        Scripted(handler).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(
            concurrency=4,
            retries=0,
            retry_backoff=0,
            budget=Budget(max_requests=2),
        ),
    )
    results = await session.fork([Query(tail=str(i), output=Answer, id=f"q{i}") for i in range(4)])
    assert calls == 2
    assert sum(result.failure is None for result in results) == 2
    assert sum(result.failure is not None and result.failure.code == "budget_exceeded" for result in results) == 2

    def allow(_report: BudgetReport):  # type: ignore[no-untyped-def]
        return Accept()

    calls = 0
    session = build(
        Scripted(handler).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(
            concurrency=1,
            retries=0,
            retry_backoff=0,
            budget=Budget(max_requests=1),
            budget_policy=allow,
        ),
    )
    results = await session.fork([Query(tail=str(i), output=Answer, id=f"q{i}") for i in range(2)])
    assert calls == 2
    assert all(result.failure is None for result in results)

    def deny(report: BudgetReport):  # type: ignore[no-untyped-def]
        return Fail(f"stop: {report.message}")

    session = build(
        Scripted(handler).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0, budget=Budget(max_requests=0), budget_policy=deny),
    )
    blocked = await session.ask(Query(tail="x", output=Answer, id="z"))
    assert blocked.failure is not None
    assert blocked.failure.message.startswith("stop:")


@pytest.mark.asyncio
async def test_usage_and_cache_details_are_aggregated() -> None:
    def handler(_messages, info, n):  # type: ignore[no-untyped-def]
        if n == 1:
            return tool_result(
                info,
                "Answer",
                {"text": "a"},
                token_usage=usage(input_tokens=100, output_tokens=10, cache_read_tokens=90),
            )
        return tool_result(
            info,
            "Answer",
            {"text": "b"},
            token_usage=usage(
                input_tokens=50,
                output_tokens=5,
                cache_read_tokens=0,
                details={"prompt_cache_hit_tokens": 40},
            ),
        )

    session = build(
        Scripted(handler).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    results = await session.fork(
        [
            Query(tail="one", output=Answer, id="a"),
            Query(tail="two", output=Answer, id="b"),
        ]
    )
    assert results[0].cache_hit_tokens == 90
    assert results[1].cache_hit_tokens == 40
    summary = forkreduce_summary(results)
    assert summary.input_tokens == 150
    assert summary.output_tokens == 15
    assert summary.cache_hit_tokens == 130
    assert summary.cache_hit_rate == pytest.approx(130 / 150)
    assert summary.by_round[0].requests == 2


def forkreduce_summary(results):  # type: ignore[no-untyped-def]
    from forkreduce.stats import summarize

    return summarize(results)
