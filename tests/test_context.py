"""Context window: local estimate warns, provider overflow fails."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from tests.support import Scripted, tool_result, user_text

from forkreduce import Query, build
from forkreduce.context import (
    ContextEstimateWarning,
    effective_context_reserve,
    estimate_tokens,
    prefix_text,
)
from forkreduce.failures import FailureKind, classify_exception
from forkreduce.policy import Policy


class Answer(BaseModel):
    """A short answer."""

    text: str


def _answer(_messages, info, _n, text="ok"):  # type: ignore[no-untyped-def]
    return tool_result(info, "Answer", {"text": text})


@contextmanager
def _always_warn() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        yield


def test_estimate_is_ceil_chars_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_default_reserve_and_override() -> None:
    assert effective_context_reserve(Policy()) == 0
    assert effective_context_reserve(Policy(context_limit=8192)) == 4096
    assert effective_context_reserve(Policy(context_limit=128_000)) == 32_000
    assert effective_context_reserve(Policy(context_limit=2000)) == 2000
    assert effective_context_reserve(Policy(context_limit=8192, context_reserve=0)) == 0
    assert effective_context_reserve(Policy(context_limit=8192, context_reserve=100)) == 100
    with pytest.raises(ValueError, match="context_reserve"):
        Policy(context_reserve=-1)
    with pytest.raises(ValueError, match="context_limit"):
        Policy(context_limit=-1)


def test_oversize_prefix_warns_and_still_builds(caplog: pytest.LogCaptureFixture) -> None:
    documents = "PREFIX-MARKER-" + ("n" * 400)
    instructions = "rules"
    with _always_warn(), warnings.catch_warnings():
        warnings.simplefilter("error", ContextEstimateWarning)
        probe = build(
            Scripted(_answer).model(),
            documents=documents,
            instructions=instructions,
            outputs=[Answer],
            warmup=False,
            policy=Policy(context_limit=10**9, context_reserve=0),
        )
    expected = estimate_tokens(prefix_text(probe.instructions, probe.rendered_documents, probe.tool_specs))
    assert probe.context_warning is None
    assert "PREFIX-MARKER-" in probe.rendered_documents
    assert "n" * 400 in probe.rendered_documents

    limit = expected + 100
    automatic = effective_context_reserve(Policy(context_limit=limit))
    with (
        _always_warn(),
        caplog.at_level(logging.WARNING, logger="forkreduce.context"),
        pytest.warns(ContextEstimateWarning, match="prefix tokens") as recorded,
    ):
        warned = build(
            Scripted(_answer).model(),
            documents=documents,
            instructions=instructions,
            outputs=[Answer],
            warmup=False,
            policy=Policy(context_limit=limit),
        )
    assert warned.prefix_hash == probe.prefix_hash
    assert warned.context_warning is not None
    assert warned.context_warning.where == "prefix"
    assert warned.context_warning.estimated_tokens == expected
    assert warned.context_warning.limit == limit
    assert warned.context_warning.reserve == automatic
    assert expected > limit - automatic
    assert str(expected) in str(recorded[0].message)
    assert str(limit) in str(recorded[0].message)
    assert str(automatic) in str(recorded[0].message)
    assert any(str(expected) in record.message for record in caplog.records)

    with _always_warn(), warnings.catch_warnings():
        warnings.simplefilter("error", ContextEstimateWarning)
        quiet = build(
            Scripted(_answer).model(),
            documents=documents,
            instructions=instructions,
            outputs=[Answer],
            warmup=False,
            policy=Policy(context_limit=limit, context_reserve=0),
        )
    assert quiet.context_warning is None
    assert quiet.prefix_hash == probe.prefix_hash

    usable_reserve = limit - expected
    with _always_warn(), pytest.warns(ContextEstimateWarning, match="prefix tokens"):
        tight = build(
            Scripted(_answer).model(),
            documents=documents,
            instructions=instructions,
            outputs=[Answer],
            warmup=False,
            policy=Policy(context_limit=limit, context_reserve=usable_reserve + 1),
        )
    assert tight.context_warning is not None
    assert tight.context_warning.reserve == usable_reserve + 1


@pytest.mark.asyncio
async def test_run_report_keeps_the_prefix_warning() -> None:
    documents = "n" * 400
    instructions = "rules"
    probe = build(
        Scripted(_answer).model(),
        documents=documents,
        instructions=instructions,
        outputs=[Answer],
        warmup=False,
        policy=Policy(context_limit=10**9, context_reserve=0, retries=0, retry_backoff=0),
    )
    expected = estimate_tokens(prefix_text(probe.instructions, probe.rendered_documents, probe.tool_specs))
    with _always_warn(), pytest.warns(ContextEstimateWarning, match="prefix tokens"):
        session = build(
            Scripted(_answer).model(),
            documents=documents,
            instructions=instructions,
            outputs=[Answer],
            warmup=False,
            policy=Policy(context_limit=expected, context_reserve=1, retries=0, retry_backoff=0),
        )
    assert session.context_warning is not None
    with _always_warn(), pytest.warns(ContextEstimateWarning):
        report = await session.map([Query(tail="hi", output=Answer, id="q")]).run()
    assert report.context_warning is session.context_warning
    assert report.context_warning.where == "prefix"
    assert report.results[0].failure is None
    assert report.results[0].output is not None
    assert any(item.where == "prefix" for item in report.context_warnings)


@pytest.mark.asyncio
async def test_oversize_tail_warns_and_still_sends() -> None:
    scripted = Scripted(_answer)
    with _always_warn(), warnings.catch_warnings():
        warnings.simplefilter("error", ContextEstimateWarning)
        session = build(
            scripted.model(),
            documents="doc",
            instructions="rules",
            outputs=[Answer],
            warmup=False,
            policy=Policy(retries=2, retry_backoff=0, context_limit=10**9, context_reserve=0),
        )
    short = Query(tail="hi", output=Answer, id="short")
    session.policy.context_limit = session.estimate_tokens(short)
    session.policy.context_reserve = 0
    with _always_warn(), warnings.catch_warnings():
        warnings.simplefilter("error", ContextEstimateWarning)
        ok = await session.ask(short)
    assert ok.failure is None
    assert ok.context_warning is None
    assert ok.output is not None and ok.output.text == "ok"
    assert len(scripted.calls) == 1

    huge = Query(tail="z" * 10000, output=Answer, id="huge")
    with _always_warn(), pytest.warns(ContextEstimateWarning, match="exceeds context limit"):
        sent = await session.ask(huge)
    assert sent.failure is None
    assert sent.output is not None and sent.output.text == "ok"
    assert sent.attempts == 1
    assert sent.context_warning is not None
    assert sent.context_warning.where == "request"
    assert sent.context_warning.estimated_tokens > sent.context_warning.limit
    assert sent.context_warning.limit == session.policy.context_limit
    assert sent.context_warning.reserve == 0
    assert len(scripted.calls) == 2
    assert "z" * 10000 in user_text(scripted.calls[1][0])


@pytest.mark.asyncio
async def test_fork_sends_an_oversize_tail() -> None:
    scripted = Scripted(_answer)
    session = build(
        scripted.model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0, context_limit=10**9, context_reserve=0),
    )
    short = Query(tail="hi", output=Answer, id="short")
    session.policy.context_limit = session.estimate_tokens(short)
    with _always_warn(), pytest.warns(ContextEstimateWarning):
        results = await session.fork(
            [
                short,
                Query(tail="z" * 10000, output=Answer, id="huge"),
            ]
        )
    assert results[0].failure is None and results[0].context_warning is None
    assert results[1].failure is None
    assert results[1].context_warning is not None
    assert results[1].context_warning.where == "request"
    assert len(scripted.calls) == 2
    with _always_warn(), pytest.warns(ContextEstimateWarning):
        report = await session.map([Query(tail="z" * 10000, output=Answer, id="mapped")]).run()
    assert report.context_warning is None
    assert report.results[0].failure is None
    assert report.results[0].context_warning is not None
    assert report.context_warnings[-1] is report.results[0].context_warning


@pytest.mark.asyncio
async def test_oversize_history_warns_and_still_sends() -> None:
    scripted = Scripted(_answer)
    session = build(
        scripted.model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0, context_limit=10**9, context_reserve=0),
    )
    first = await session.ask(Query(tail="hi", output=Answer, id="first"))
    assert first.failure is None
    assert first.messages
    fitting = Query(tail="next", output=Answer, id="fit", history=first.messages)
    session.policy.context_limit = session.estimate_tokens(fitting)
    huge_history = list(first.messages)
    huge_history.append(ModelRequest(parts=[UserPromptPart(content="h" * 8000)]))
    with _always_warn(), pytest.warns(ContextEstimateWarning):
        sent = await session.ask(Query(tail="next", output=Answer, id="again", history=huge_history))
    assert sent.failure is None
    assert sent.context_warning is not None
    assert sent.context_warning.estimated_tokens > sent.context_warning.limit
    assert len(scripted.calls) == 2


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
    assert failed.context_warning is None
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
