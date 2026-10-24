"""Native feedback accounting across citation repairs and reduce continuations."""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo
from tests.support import Scripted, text_result, tool_result

from forkreduce import (
    Accept,
    Citation,
    CitationCheck,
    CitationRepair,
    DiscardExplanations,
    FailureKind,
    Policy,
    Query,
    RequireReason,
    Traced,
    build,
)
from forkreduce.coverage import ReduceReport
from forkreduce.documents import Block
from forkreduce.reduce import ReduceItem


class Answer(Citation):
    """An answer whose citations can be repaired independently of its text."""

    text: str


class Finding(Traced):
    text: str


class Findings(BaseModel):
    """A traced reduce output."""

    items: list[Finding]


DOCUMENT = "The library opens at nine."


def response(info: AgentInfo, name: str, payload: dict[str, Any]) -> ModelResponse:
    return tool_result(info, name, payload) if info.output_tools else text_result(payload)


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("ending", ["repaired", "needs_review", "validation_failure"])
@pytest.mark.asyncio
async def test_check_repair_accumulates_feedback_from_every_round(
    mode: Literal["tools", "prompted"], ending: str
) -> None:
    def handler(_messages: list[ModelMessage], info: AgentInfo, n: int) -> ModelResponse:
        # Each query needs one native validation retry, including the last
        # repair when that repair ultimately exhausts its feedback allowance.
        payload: dict[str, Any] = {}
        if n % 2 == 0 and not (n == 6 and ending == "validation_failure"):
            payload = {
                "text": "original" if n == 2 else "must not replace original",
                "quote": DOCUMENT if n == 6 and ending == "repaired" else "missing passage",
                "block_id": "hours",
            }
        return response(info, "Answer", payload)

    scripted = Scripted(handler)
    session = build(
        scripted.model(),
        documents=Block(block_id="hours", text=DOCUMENT),
        instructions="Cite the opening hours.",
        outputs=[Answer],
        output_mode=mode,
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0, output_feedback_retries=1),
    )
    run = await (
        session.map([Query(tail="When does it open?", output=Answer, id="hours", round=3)])
        .check(CitationCheck(), repair=CitationRepair(max_rounds=2))
        .run()
    )

    assert len(run.results) == 1
    result = run.results[0]
    assert result.query_id == "hours"
    assert result.output == Answer(
        text="original",
        quote=DOCUMENT if ending == "repaired" else "missing passage",
        block_id="hours",
    )
    assert result.feedback_retries_used == 3
    assert result.meta == {"feedback_retries_used": 3}
    assert result.attempts == 3
    assert result.usage.requests == run.usage.requests == len(scripted.calls) == 6
    assert (result.usage.input_tokens, result.usage.output_tokens, result.cache_hit_tokens) == (600, 60, 480)
    assert (run.usage.input_tokens, run.usage.output_tokens, run.usage.cache_hit_tokens) == (600, 60, 480)
    assert result.prefix_hash == session.prefix_hash
    assert result.resolved_output_mode == mode
    assert result.needs_review is (ending != "repaired")
    assert result.failure is None
    report = run.rounds[0].check_reports[0]
    assert report.repair_rounds == 2
    assert report.action == ("repaired" if ending == "repaired" else "needs_review")
    if ending == "validation_failure":
        assert len(run.failures) == 1
        assert run.failures[0].kind is FailureKind.VALIDATION
    else:
        assert run.failures == []


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.asyncio
async def test_reduce_combines_feedback_without_recounting_continuation_history(
    mode: Literal["tools", "prompted"], complete: bool
) -> None:
    def handler(messages: list[ModelMessage], info: AgentInfo, n: int) -> ModelResponse:
        if n <= 2:
            payload = {} if n == 1 else {"items": [{"text": "merged", "source_ids": ["a"]}]}
            return response(info, "Findings", payload)
        # Both reason queries continue the reduce branch. Their own native
        # feedback is additional work, not a replay of the merge's feedback.
        assert any(isinstance(message, ModelResponse) for message in messages)
        payload = {}
        if n % 2 == 0:
            payload = {"explanations": [{"item_id": "b", "reason": "duplicate of a"}] if n == 6 and complete else []}
        return response(info, "DiscardExplanations", payload)

    class Explain:
        def decide(self, report: ReduceReport) -> Accept | RequireReason:
            return RequireReason(tuple(report.uncovered)) if report.uncovered else Accept()

    scripted = Scripted(handler)
    session = build(
        scripted.model(),
        documents=DOCUMENT,
        instructions="Merge the supplied findings.",
        outputs=[Findings, DiscardExplanations],
        output_mode=mode,
        warmup=False,
        policy=Policy(retries=1, retry_backoff=0, output_feedback_retries=1),
    )
    outcome = await session.reduce(
        [ReduceItem(id="a", value="opening hours"), ReduceItem(id="b", value="same opening hours")],
        output=Findings,
        coverage=Explain(),
        round_index=2,
    )

    assert outcome.action == "RequireReason"
    assert outcome.output == Findings(items=[Finding(text="merged", source_ids=["a"])])
    assert len(outcome.calls) == 3
    assert [call.feedback_retries_used for call in outcome.calls] == [1, 1, 1]
    assert [call.meta for call in outcome.calls] == [{"feedback_retries_used": 1}] * 3
    assert [call.usage.requests for call in outcome.calls] == [2, 2, 2]
    assert [call.attempts for call in outcome.calls] == [1, 1, 1]
    result = outcome.result
    assert result is not None
    assert result.feedback_retries_used == 3
    assert result.meta == {"feedback_retries_used": 3}
    assert result.attempts == 3
    assert result.usage.requests == len(scripted.calls) == 6
    assert (result.usage.input_tokens, result.usage.output_tokens, result.cache_hit_tokens) == (600, 60, 480)
    assert result.query_id == outcome.calls[-1].query_id
    assert result.output == outcome.calls[-1].output
    assert result.messages == outcome.calls[-1].messages
    assert result.round == 2
    assert result.prefix_hash == session.prefix_hash
    assert result.resolved_output_mode == mode
    assert outcome.fatal is not complete
    if complete:
        assert outcome.failure is None
        assert [(reason.item_id, reason.reason) for reason in outcome.reasons] == [("b", "duplicate of a")]
    else:
        assert outcome.failure is not None
        assert outcome.failure.code == "reasons_incomplete"
        assert outcome.reasons == []
