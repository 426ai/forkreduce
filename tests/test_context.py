"""Context-window hard-fail: oversize prefix, oversize tail, provider overflow."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from tests.support import Scripted, tool_result, user_text

from forkreduce import Query, build
from forkreduce.context import estimate_tokens, prefix_text
from forkreduce.errors import ForkReduceError
from forkreduce.failures import FailureKind, classify_exception
from forkreduce.policy import Budget, Policy


class Answer(BaseModel):
    """A short answer."""

    text: str


def _answer(_messages, info, _n, text="ok"):  # type: ignore[no-untyped-def]
    return tool_result(info, "Answer", {"text": text})


def test_estimate_is_ceil_chars_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_oversize_prefix_raises_with_size_and_limit() -> None:
    documents = "PREFIX-MARKER-" + ("n" * 400)
    instructions = "rules"
    probe = build(
        Scripted(_answer).model(),
        documents=documents,
        instructions=instructions,
        outputs=[Answer],
        warmup=False,
        policy=Policy(context_limit=10**9),
    )
    expected = estimate_tokens(prefix_text(probe.instructions, probe.rendered_documents, probe.tool_specs))
    assert "PREFIX-MARKER-" in probe.rendered_documents
    assert "n" * 400 in probe.rendered_documents

    with pytest.raises(ForkReduceError) as caught:
        build(
            Scripted(_answer).model(),
            documents=documents,
            instructions=instructions,
            outputs=[Answer],
            warmup=False,
            policy=Policy(context_limit=expected - 1),
        )
    assert caught.value.code == "context_overflow"
    assert str(caught.value) == (
        f"context overflow: estimated {expected} tokens exceeds context limit {expected - 1} (prefix)"
    )

    fitting = build(
        Scripted(_answer).model(),
        documents=documents,
        instructions=instructions,
        outputs=[Answer],
        warmup=False,
        policy=Policy(context_limit=expected),
    )
    assert fitting.prefix_hash == probe.prefix_hash
    assert "n" * 400 in fitting.rendered_documents


def test_negative_context_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="context_limit"):
        Policy(context_limit=-1)


@pytest.mark.asyncio
async def test_oversize_tail_fails_before_the_model_and_does_not_reserve() -> None:
    scripted = Scripted(_answer)
    session = build(
        scripted.model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(
            retries=2,
            retry_backoff=0,
            context_limit=10**9,
            budget=Budget(max_requests=5),
        ),
    )
    short = Query(tail="hi", output=Answer, id="short")
    session.policy.context_limit = session.estimate_tokens(short)
    ok = await session.ask(short)
    assert ok.failure is None
    assert ok.output is not None and ok.output.text == "ok"
    assert len(scripted.calls) == 1
    assert session._budget.requests == 1

    huge = Query(tail="z" * 10000, output=Answer, id="huge")
    failed = await session.ask(huge)
    assert failed.output is None
    assert failed.attempts == 0
    assert failed.failure is not None
    assert failed.failure.code == "context_overflow"
    assert failed.failure.kind is FailureKind.PERMANENT
    assert failed.failure.retryable is False
    estimated = failed.failure.detail["estimated_tokens"]
    limit = failed.failure.detail["limit"]
    assert estimated > limit
    assert failed.failure.detail["where"] == "request"
    assert failed.failure.message == (
        f"context overflow: estimated {estimated} tokens exceeds context limit {limit} (request)"
    )
    assert len(scripted.calls) == 1
    assert session._budget.requests == 1
    assert "z" * 10000 not in user_text(scripted.calls[0][0])


@pytest.mark.asyncio
async def test_fork_keeps_a_fitting_query_when_another_tail_overflows() -> None:
    scripted = Scripted(_answer)
    session = build(
        scripted.model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=2, retry_backoff=0, context_limit=10**9),
    )
    short = Query(tail="hi", output=Answer, id="short")
    session.policy.context_limit = session.estimate_tokens(short)
    results = await session.fork(
        [
            short,
            Query(tail="z" * 10000, output=Answer, id="huge"),
        ]
    )
    assert results[0].failure is None
    assert results[1].failure is not None
    assert results[1].failure.code == "context_overflow"
    assert results[1].attempts == 0
    assert len(scripted.calls) == 1


@pytest.mark.asyncio
async def test_oversize_history_fails_without_a_second_model_call() -> None:
    scripted = Scripted(_answer)
    session = build(
        scripted.model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=2, retry_backoff=0, context_limit=10**9),
    )
    first = await session.ask(Query(tail="hi", output=Answer, id="first"))
    assert first.failure is None
    assert first.messages
    fitting = Query(tail="next", output=Answer, id="fit", history=first.messages)
    session.policy.context_limit = session.estimate_tokens(fitting)
    huge_history = list(first.messages)
    huge_history.append(ModelRequest(parts=[UserPromptPart(content="h" * 8000)]))
    failed = await session.ask(Query(tail="next", output=Answer, id="again", history=huge_history))
    assert failed.attempts == 0
    assert failed.failure is not None
    assert failed.failure.code == "context_overflow"
    assert failed.failure.detail["estimated_tokens"] > failed.failure.detail["limit"]
    assert len(scripted.calls) == 1


@pytest.mark.asyncio
async def test_provider_context_overflow_is_permanent() -> None:
    def denied(_messages, _info, _n):  # type: ignore[no-untyped-def]
        raise ModelHTTPError(
            status_code=400,
            model_name="m",
            body="This model's maximum context length is 8192 tokens. However, you requested 9000 tokens.",
        )

    scripted = Scripted(denied)
    session = build(
        scripted.model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=2, retry_backoff=0),
    )
    failed = await session.ask(Query(tail="go", output=Answer, id="q"))
    assert failed.attempts == 1
    assert len(scripted.calls) == 1
    assert failed.failure is not None
    assert failed.failure.code == "context_overflow"
    assert failed.failure.retryable is False
    assert failed.failure.kind is FailureKind.PERMANENT
    assert failed.failure.detail["reported_tokens"] == 9000
    assert failed.failure.detail["limit"] == 8192
    assert failed.failure.detail["status_code"] == 400
    assert failed.failure.detail["where"] == "provider"
    assert "reported 9000 tokens exceeds context limit 8192" in failed.failure.message


def test_provider_overflow_marker_without_figures_still_classifies() -> None:
    failure = classify_exception(ModelHTTPError(status_code=400, model_name="m", body="context_length_exceeded"))
    assert failure.code == "context_overflow"
    assert failure.retryable is False
    assert "unknown" in failure.message
    ordinary = classify_exception(ModelHTTPError(status_code=400, model_name="m", body="bad request"))
    assert ordinary.code == "http_400"
