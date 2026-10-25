"""Native output-validation feedback, prefix stability, and per-request budgets."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from forkreduce import Accept, Budget, BudgetReport, FailureKind, OutputValidator, Policy, Query, Session, build


class Answer(BaseModel):
    """A bounded synthetic answer."""

    model_config = ConfigDict(extra="forbid")
    value: int
    low: int = 0
    high: int = 1

    @model_validator(mode="after")
    def ordered(self) -> Answer:
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        return self


class Note(BaseModel):
    """Another registered output."""

    text: str


def positive(output: Any) -> Any:
    if isinstance(output, Answer) and output.value <= 0:
        raise ModelRetry("value must be positive")
    return output


async def positive_async(output: Any) -> Any:
    return positive(output)


class RecordingModel:
    def __init__(self, payload: Callable[[int], dict[str, Any]], *, output_name: str = "Answer") -> None:
        self.payload = payload
        self.output_name = output_name
        self.calls: list[tuple[list[ModelMessage], AgentInfo]] = []
        self.model = FunctionModel(self.respond)

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(info)))
        payload = self.payload(len(self.calls))
        part = (
            ToolCallPart(self.output_name, payload)
            if info.output_tools
            else TextPart(json.dumps(payload, ensure_ascii=False))
        )
        return ModelResponse(
            parts=[part],
            usage=RequestUsage(input_tokens=10, output_tokens=3, cache_read_tokens=4),
        )


def session_for(
    recorded: RecordingModel,
    *,
    mode: str = "tools",
    feedback: int = 1,
    validator: OutputValidator | None = None,
    policy: Policy | None = None,
) -> Session:
    return build(
        recorded.model,
        documents="α shared synthetic evidence",
        instructions="Return a grounded synthetic answer.",
        outputs=[Answer, Note],
        output_mode=mode,  # type: ignore[arg-type]
        output_validator=validator,
        warmup=False,
        policy=policy or Policy(retries=0, retry_backoff=0, output_feedback_retries=feedback),
    )


def prefix_bytes(messages: list[ModelMessage]) -> bytes:
    first = messages[0]
    assert isinstance(first, ModelRequest)
    part = first.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, str)
    return part.content.encode("utf-8")


def feedback_text(messages: list[ModelMessage]) -> str:
    # Tool and prompted modes encode native retry feedback differently across
    # SDK versions. Inspect requests after the rejected response, not user tails.
    feedback: list[str] = []
    response_seen = False
    for message in messages:
        if isinstance(message, ModelResponse):
            response_seen = True
        elif response_seen and isinstance(message, ModelRequest):
            feedback.extend(str(part.content) for part in message.parts if hasattr(part, "content"))
    assert feedback, "the model must receive validation feedback after its rejected response"
    return "\n".join(feedback)


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize(
    ("invalid", "validator", "error"),
    [
        ({"value": "not-an-integer"}, None, "value"),
        ({"value": 1, "low": 9, "high": 2}, None, "low must not exceed high"),
        ({"value": 0}, positive, "value must be positive"),
        ({"value": 0}, positive_async, "value must be positive"),
    ],
)
@pytest.mark.asyncio
async def test_native_feedback_reuses_branch_and_frozen_prefix(
    mode: str, invalid: dict[str, Any], validator: OutputValidator | None, error: str
) -> None:
    recorded = RecordingModel(lambda n: invalid if n == 1 else {"value": 7})
    session = session_for(recorded, mode=mode, validator=validator)
    baseline = session_for(recorded, mode=mode, feedback=0)
    result = await session.ask(Query(tail="Give the answer.", output=Answer, id="answer"))

    assert result.output == Answer(value=7)
    assert result.failure is None
    assert result.attempts == 1
    assert result.feedback_retries_used == 1
    assert result.meta == {"feedback_retries_used": 1}
    assert result.usage.requests == len(recorded.calls) == 2
    assert (result.usage.input_tokens, result.usage.output_tokens, result.cache_hit_tokens) == (20, 6, 8)
    assert (session._budget.input_tokens, session._budget.output_tokens) == (20, 6)
    first_messages, first_info = recorded.calls[0]
    second_messages, second_info = recorded.calls[1]
    assert error in feedback_text(second_messages)
    assert any(isinstance(message, ModelResponse) for message in second_messages)
    assert prefix_bytes(first_messages) == prefix_bytes(second_messages) == session.rendered_documents.encode("utf-8")
    assert first_info.output_tools == second_info.output_tools
    assert first_info.instructions == second_info.instructions == session.instructions
    assert [tool.name for tool in first_info.output_tools] == (["Answer", "Note"] if mode == "tools" else [])
    assert session.prefix_hash == baseline.prefix_hash
    assert session.tool_specs == baseline.tool_specs
    assert session.starts_with_prefix(result.messages)
    if mode == "prompted":
        assert "properties" not in (first_info.instructions or "")
        assert "properties" not in (second_info.instructions or "")


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.asyncio
async def test_default_zero_keeps_single_request_and_unchanged_prefix(mode: str) -> None:
    recorded = RecordingModel(lambda _n: {"value": "invalid"})
    session = session_for(recorded, mode=mode, feedback=0)
    result = await session.ask(Query(tail="Give the answer.", output=Answer))
    assert len(recorded.calls) == result.usage.requests == 1
    assert result.failure is not None
    assert result.failure.kind is FailureKind.VALIDATION
    assert result.feedback_retries_used == 0
    assert result.meta == {"feedback_retries_used": 0}
    default_messages, default_info = recorded.calls[0]

    enabled = session_for(recorded, mode=mode)
    await enabled.ask(Query(tail="Give the answer.", output=Answer))
    enabled_messages, enabled_info = recorded.calls[1]
    assert prefix_bytes(default_messages) == prefix_bytes(enabled_messages)
    assert default_info.output_tools == enabled_info.output_tools
    assert default_info.instructions == enabled_info.instructions
    assert session.render_tail(Query(tail="Give the answer.", output=Answer)) == enabled.render_tail(
        Query(tail="Give the answer.", output=Answer)
    )
    assert session.prefix_hash == enabled.prefix_hash


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("validator", [positive, positive_async])
@pytest.mark.asyncio
async def test_validator_with_zero_feedback_still_validates_without_second_request(
    mode: str, validator: OutputValidator
) -> None:
    recorded = RecordingModel(lambda _n: {"value": 0})
    session = session_for(recorded, mode=mode, feedback=0, validator=validator)
    assert session.output_validator is validator
    result = await session.ask(Query(tail="Validate without feedback.", output=Answer))
    assert len(recorded.calls) == 1
    assert result.failure is not None
    assert result.failure.kind is FailureKind.VALIDATION
    assert result.feedback_retries_used == 0
    assert result.meta == {"feedback_retries_used": 0}
    assert result.usage.requests == 1


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("feedback", [1, 2])
@pytest.mark.parametrize("validator", [None, positive, positive_async])
@pytest.mark.asyncio
async def test_exhausted_feedback_bound_records_failure_usage(
    mode: str, feedback: int, validator: OutputValidator | None
) -> None:
    recorded = RecordingModel(lambda _n: {"value": 0} if validator else {"value": "invalid"})
    session = session_for(recorded, mode=mode, feedback=feedback, validator=validator)
    result = await session.run_once(Query(tail="Never valid.", output=Answer))
    assert result.failure is not None
    assert result.failure.kind is FailureKind.VALIDATION
    assert result.output is None
    assert result.attempts == 1
    assert result.feedback_retries_used == feedback
    assert result.meta == {"feedback_retries_used": feedback}
    assert len(recorded.calls) == result.usage.requests == feedback + 1
    assert (result.usage.input_tokens, result.usage.output_tokens) == (10 * (feedback + 1), 3 * (feedback + 1))
    assert (session._budget.input_tokens, session._budget.output_tokens) == (
        result.usage.input_tokens,
        result.usage.output_tokens,
    )


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize(
    "budget",
    [
        Budget(max_requests=1),
        Budget(max_input_tokens=10),
        Budget(max_output_tokens=3),
        Budget(max_total_tokens=13),
    ],
)
@pytest.mark.asyncio
async def test_budgets_prevent_feedback_request_after_first_billed_response(mode: str, budget: Budget) -> None:
    recorded = RecordingModel(lambda n: {"value": "invalid"} if n == 1 else {"value": 7})
    session = session_for(recorded, mode=mode, policy=Policy(retries=2, output_feedback_retries=3, budget=budget))
    result = await session.ask(Query(tail="Budgeted answer.", output=Answer))
    assert len(recorded.calls) == result.usage.requests == 1
    assert result.failure is not None
    assert result.failure.kind is FailureKind.BUDGET
    assert result.failure.code == "budget_exceeded"
    assert result.attempts == 1
    assert result.feedback_retries_used == 0
    assert result.meta == {"feedback_retries_used": 0}
    assert session._budget.requests == 1
    assert (result.usage.input_tokens, result.usage.output_tokens) == (10, 3)
    assert (session._budget.input_tokens, session._budget.output_tokens) == (10, 3)


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.asyncio
async def test_budget_rejection_after_one_feedback_preserves_used_count(mode: str) -> None:
    recorded = RecordingModel(lambda _n: {"value": "invalid"})
    session = session_for(
        recorded,
        mode=mode,
        policy=Policy(retries=1, output_feedback_retries=3, budget=Budget(max_requests=2)),
    )
    result = await session.ask(Query(tail="Two requests only.", output=Answer))
    assert result.failure is not None
    assert result.failure.kind is FailureKind.BUDGET
    assert len(recorded.calls) == result.usage.requests == session._budget.requests == 2
    assert result.attempts == 1
    assert result.feedback_retries_used == 1
    assert result.meta == {"feedback_retries_used": 1}


@pytest.mark.asyncio
async def test_budget_accept_override_applies_to_later_feedback_requests() -> None:
    reports: list[BudgetReport] = []

    def allow(report: BudgetReport) -> Accept:
        reports.append(report)
        return Accept()

    recorded = RecordingModel(lambda n: {"value": "invalid"} if n == 1 else {"value": 7})
    session = session_for(
        recorded,
        policy=Policy(
            retries=0,
            output_feedback_retries=1,
            budget=Budget(max_total_tokens=13),
            budget_policy=allow,
        ),
    )
    result = await session.ask(Query(tail="Allow another request.", output=Answer))
    assert result.output == Answer(value=7)
    assert len(recorded.calls) == result.usage.requests == session._budget.requests == 2
    assert result.feedback_retries_used == 1
    assert len(reports) == 1
    assert (reports[0].requests, reports[0].input_tokens, reports[0].output_tokens) == (1, 10, 3)


@pytest.mark.parametrize("succeed", [False, True])
@pytest.mark.asyncio
async def test_feedback_counts_aggregate_across_outer_retries(succeed: bool) -> None:
    recorded = RecordingModel(lambda n: {"value": 7} if succeed and n == 4 else {"value": "invalid"})
    session = session_for(
        recorded,
        policy=Policy(retries=1, retry_backoff=0, output_feedback_retries=1, budget=Budget(max_requests=4)),
    )
    result = await session.ask(Query(tail="Outer retry.", output=Answer))
    assert (result.failure is None) is succeed
    assert result.attempts == 2
    assert result.feedback_retries_used == 2
    assert result.meta == {"feedback_retries_used": 2}
    assert len(recorded.calls) == result.usage.requests == session._budget.requests == 4
    assert (result.usage.input_tokens, result.usage.output_tokens, result.cache_hit_tokens) == (40, 12, 16)
    assert (session._budget.input_tokens, session._budget.output_tokens) == (40, 12)


@pytest.mark.asyncio
async def test_feedback_budget_denial_before_first_call_records_zero() -> None:
    recorded = RecordingModel(lambda _n: {"value": 7})
    session = session_for(recorded, policy=Policy(output_feedback_retries=2, budget=Budget(max_requests=0)))
    result = await session.ask(Query(tail="No request permitted.", output=Answer))
    assert recorded.calls == []
    assert result.failure is not None
    assert result.failure.kind is FailureKind.BUDGET
    assert result.attempts == result.usage.requests == result.feedback_retries_used == 0
    assert result.meta == {"feedback_retries_used": 0}


@pytest.mark.asyncio
async def test_prompted_validator_uses_each_queries_expected_type() -> None:
    recorded = RecordingModel(lambda _n: {"value": 7})
    session = session_for(recorded, mode="prompted", validator=positive_async)
    first = await session.ask(Query(tail="Answer.", output=Answer))
    recorded.payload = lambda _n: {"text": "A synthetic note."}
    second = await session.ask(Query(tail="Note.", output=Note))
    assert first.output == Answer(value=7)
    assert second.output == Note(text="A synthetic note.")
    assert first.feedback_retries_used == second.feedback_retries_used == 0
    assert len(recorded.calls) == 2
    assert prefix_bytes(recorded.calls[0][0]) == prefix_bytes(recorded.calls[1][0])
    assert recorded.calls[0][1].instructions == recorded.calls[1][1].instructions


@pytest.mark.asyncio
async def test_continuation_usage_does_not_rebill_prior_history() -> None:
    recorded = RecordingModel(lambda n: {"value": "invalid"} if n in {1, 3} else {"value": 7})
    session = session_for(recorded, policy=Policy(retries=0, output_feedback_retries=1, budget=Budget(max_requests=4)))
    first = await session.ask(Query(tail="First answer.", output=Answer))
    second = await session.ask(Query(tail="Continue.", output=Answer, history=first.messages))
    assert first.failure is second.failure is None
    assert first.usage.requests == second.usage.requests == 2
    assert first.usage.input_tokens == second.usage.input_tokens == 20
    assert first.feedback_retries_used == second.feedback_retries_used == 1
    assert session._budget.requests == 4
    assert (session._budget.input_tokens, session._budget.output_tokens) == (40, 12)


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.asyncio
async def test_feedback_transport_errors_count_actual_requests(recover: bool) -> None:
    def payload(n: int) -> dict[str, Any]:
        if n == 2:
            raise ConnectionError("Synthetic offline transport error")
        return {"value": 7} if n == 4 else {"value": "invalid"}

    recorded = RecordingModel(payload)
    session = session_for(
        recorded,
        policy=Policy(
            retries=1 if recover else 0,
            retry_backoff=0,
            output_feedback_retries=1,
            budget=Budget(max_requests=4),
        ),
    )
    result = await session.ask(Query(tail="Transport during feedback.", output=Answer))
    assert (result.failure is None) is recover
    if not recover:
        assert result.failure is not None
        assert result.failure.kind is FailureKind.TRANSPORT
    expected_requests = 4 if recover else 2
    assert len(recorded.calls) == result.usage.requests == session._budget.requests == expected_requests
    assert result.attempts == (2 if recover else 1)
    assert result.feedback_retries_used == (2 if recover else 1)
    assert result.meta == {"feedback_retries_used": result.feedback_retries_used}
    # The failed transport has a request slot, but no billed response.
    assert (result.usage.input_tokens, result.usage.output_tokens) == (
        10 * (expected_requests - 1),
        3 * (expected_requests - 1),
    )
    assert (session._budget.input_tokens, session._budget.output_tokens) == (
        result.usage.input_tokens,
        result.usage.output_tokens,
    )


@pytest.mark.asyncio
async def test_result_metadata_is_safe_and_independent() -> None:
    recorded = RecordingModel(lambda _n: {"value": 7})
    session = session_for(recorded)
    metadata = {"query_body": "Synthetic caller-only content"}
    result = await session.ask(Query(tail="Synthetic private query.", output=Answer, meta=metadata))
    assert result.meta == {"feedback_retries_used": 0}
    result.meta["feedback_retries_used"] = 99
    another = await session.ask(Query(tail="Another query.", output=Answer))
    assert another.meta == {"feedback_retries_used": 0}
    assert metadata == {"query_body": "Synthetic caller-only content"}


def test_feedback_retry_policy_is_nonnegative() -> None:
    with pytest.raises(ValueError, match="output_feedback_retries must be >= 0"):
        Policy(output_feedback_retries=-1)
    assert Policy().output_feedback_retries == 0
