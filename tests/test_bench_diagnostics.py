"""Bounded opt-in failure diagnostics using synthetic responses only."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from benchmarks.budget import BudgetExceeded, BudgetHalted
from benchmarks.diagnostics import MAX_RESPONSE_CHARS
from benchmarks.methods import METHODS
from benchmarks.providers import RecordingModel
from benchmarks.schema import Answers, Contract, MethodConfig, MethodResult, Question
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

PAYLOAD = {"answers": [{"question_id": "q", "spans": [], "source_ids": ["fork:0"]}]}
PARAMETERS = ModelRequestParameters(
    output_tools=[ToolDefinition(name="Answers", parameters_json_schema=Answers.model_json_schema())]
)
MESSAGES = [ModelRequest(parts=[UserPromptPart("PROMPT_MUST_NOT_BE_CAPTURED")])]


class ResponseModel(TestModel):
    def __init__(self, response: ModelResponse, error: Exception | None = None) -> None:
        super().__init__()
        self.response = response
        self.error = error

    async def request(self, messages: Any, model_settings: Any, model_request_parameters: Any) -> ModelResponse:
        if self.error is not None:
            raise self.error
        return self.response


def _response(payload: Any = None, **kwargs: Any) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart("Answers", PAYLOAD if payload is None else payload)],
        usage=RequestUsage(input_tokens=100, output_tokens=20, details={"reasoning_tokens": 7}),
        **kwargs,
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert all(len(line) <= MAX_RESPONSE_CHARS for line in lines)
    return [json.loads(line) for line in lines]


async def test_diagnostics_default_off_and_opt_in_success_creates_no_file(tmp_path: Path) -> None:
    wrapped = ResponseModel(_response())
    default = RecordingModel(wrapped, MethodResult())
    await default.request(MESSAGES, None, PARAMETERS)
    default.capture_failure("validation")
    assert list(tmp_path.iterdir()) == []
    assert default._response.get() is None

    destination = tmp_path / "nested" / "failures.jsonl"
    enabled = RecordingModel(wrapped, MethodResult(), failure_path=destination)
    await enabled.request(MESSAGES, None, PARAMETERS)
    assert not destination.exists()
    assert enabled.result.usage[0]["finish_reason"] is None
    assert enabled.result.usage[0]["tool_call_emitted"] is True
    assert enabled.result.usage[0]["tool_names"] == ["Answers"]


async def test_only_failed_attempt_is_saved_and_duplicate_hooks_do_not_repeat_it(tmp_path: Path) -> None:
    destination = tmp_path / "failed.jsonl"
    model = ResponseModel(_response('{"answers":'))
    recorder = RecordingModel(model, MethodResult(), failure_path=str(destination))
    failed = await recorder.request(MESSAGES, None, PARAMETERS)
    recorder.capture_failure("validation")
    recorder.capture_failure("validation", response=failed)
    model.response = _response()
    await recorder.request(MESSAGES, None, PARAMETERS)
    rows = _rows(destination)
    assert len(rows) == 1
    assert rows[0]["tools"] == [{"name": "Answers", "arguments": '{"answers":'}]
    assert rows[0]["reasoning_tokens"] == 7
    assert rows[0]["stage"] == "extraction"
    assert len(recorder.result.usage) == 2


@pytest.mark.parametrize(
    "response,reason,expected,json_status,schema_status",
    [
        (_response('{"answers":', finish_reason="length"), "validation", "length", "malformed_json", "not_checked"),
        (
            ModelResponse(parts=[TextPart(json.dumps(PAYLOAD))]),
            "validation",
            "no_tool",
            "valid_json",
            "valid",
        ),
        (_response('{"answers":'), "validation", "malformed_json", "malformed_json", "not_checked"),
        (_response({"answers": "bad"}), "validation", "schema_validation", "valid_json", "invalid"),
        (_response(), "question_ids", "question_ids", "valid_json", "valid"),
        (_response(), "reduce_provenance", "reduce_provenance", "valid_json", "valid"),
        (_response(), "citation_invalid", "citation_invalid", "valid_json", "valid"),
    ],
)
async def test_failure_classifications(
    tmp_path: Path, response: ModelResponse, reason: str, expected: str, json_status: str, schema_status: str
) -> None:
    destination = tmp_path / "failed.jsonl"
    recorder = RecordingModel(ResponseModel(response), MethodResult(), failure_path=destination)
    await recorder.request(MESSAGES, None, PARAMETERS)
    recorder.capture_failure(reason)
    row = _rows(destination)[0]
    assert row["failure_reason"] == reason
    assert row["classification"] == expected
    assert row["json_classification"] == json_status
    assert row["schema_classification"] == schema_status
    assert row["tool_call_emitted"] is (expected != "no_tool")
    assert row["finish_reason"] == ("length" if expected == "length" else None)


async def test_prompted_text_is_json_classified_without_requiring_a_tool(tmp_path: Path) -> None:
    destination = tmp_path / "failed.jsonl"
    response = ModelResponse(parts=[TextPart('{"answers":')])
    recorder = RecordingModel(ResponseModel(response), MethodResult(), failure_path=destination)
    await recorder.request(MESSAGES, None, ModelRequestParameters())
    recorder.capture_failure("validation")
    row = _rows(destination)[0]
    assert row["classification"] == "malformed_json"
    assert row["text"] == '{"answers":'
    assert row["tools"] == []


async def test_redaction_excludes_thinking_requests_metadata_and_live_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHMARK_API_KEY", "live-credential-unique-913")
    monkeypatch.setenv("BENCHMARK_AUTH_TOKEN", "actual-token-with-no-key-prefix")
    destination = tmp_path / "failed.jsonl"
    response = ModelResponse(
        parts=[
            ThinkingPart("THINKING_MUST_NOT_BE_CAPTURED"),
            ToolCallPart("Answers", '{"token":"actual-token-with-no-key-prefix", "x":"live-credential-unique-913"}'),
            TextPart("sk-ant-abcdef1234567890 API_KEY=unconfigured-secret Bearer bearer-secret-123"),
        ],
        provider_details={"headers": "HEADERS_MUST_NOT_BE_CAPTURED", "prompt": "PROVIDER_PROMPT_PRIVATE"},
    )
    recorder = RecordingModel(ResponseModel(response), MethodResult(), failure_path=destination)
    await recorder.request(MESSAGES, {"extra_headers": {"Authorization": "REQUEST_HEADER_PRIVATE"}}, PARAMETERS)
    recorder.capture_failure("validation")
    raw = destination.read_text(encoding="utf-8")
    for secret in (
        "live-credential-unique-913",
        "actual-token-with-no-key-prefix",
        "sk-ant-abcdef1234567890",
        "unconfigured-secret",
        "bearer-secret-123",
        "THINKING_MUST_NOT_BE_CAPTURED",
        "PROMPT_MUST_NOT_BE_CAPTURED",
        "HEADERS_MUST_NOT_BE_CAPTURED",
        "PROVIDER_PROMPT_PRIVATE",
        "REQUEST_HEADER_PRIVATE",
    ):
        assert secret not in raw
    assert "[REDACTED]" in raw
    assert _rows(destination)[0]["tool_call_emitted"] is True


async def test_total_json_bound_covers_multiple_tools_text_and_escaped_characters(tmp_path: Path) -> None:
    destination = tmp_path / "failed.jsonl"
    response = ModelResponse(
        parts=[
            TextPart('visible "\\\n' * 600),
            *[ToolCallPart("Answers", {"answers": "x" * 3000}) for _ in range(40)],
            TextPart("end" * 3000),
        ],
        finish_reason="length",
    )
    recorder = RecordingModel(ResponseModel(response), MethodResult(), failure_path=destination)
    await recorder.request(MESSAGES, None, PARAMETERS)
    recorder.capture_failure("validation")
    rows = _rows(destination)
    assert rows[0]["truncated"] is True
    assert rows[0]["classification"] == "length"
    assert rows[0]["tool_call_emitted"] is True
    assert len(destination.read_text(encoding="utf-8")) <= MAX_RESPONSE_CHARS + 1


class PrivateProviderError(RuntimeError):
    status_code = 503

    @property
    def body(self) -> Any:
        raise AssertionError("exception bodies must not be inspected")

    def __str__(self) -> str:
        raise AssertionError("exception text must not be inspected")


@pytest.mark.parametrize("error", [PrivateProviderError("PRIVATE"), BudgetExceeded("PRIVATE"), BudgetHalted("PRIVATE")])
async def test_transport_and_budget_failures_capture_only_type_status_and_no_stale_response(
    tmp_path: Path, error: Exception
) -> None:
    destination = tmp_path / "failed.jsonl"
    model = ResponseModel(_response())
    recorder = RecordingModel(model, MethodResult(), failure_path=destination)
    previous = await recorder.request(MESSAGES, None, PARAMETERS)
    model.error = error
    with pytest.raises(type(error)):
        await recorder.request(MESSAGES, None, PARAMETERS)
    recorder.capture_failure("validation", response=previous)
    row = _rows(destination)[0]
    budget = isinstance(error, (BudgetExceeded, BudgetHalted))
    assert row == {
        "stage": "extraction",
        "failure_reason": "budget_refusal" if budget else "provider_error",
        "classification": "budget_refusal" if budget else "provider_error",
        "exception_type": type(error).__name__,
        "status_code": None if budget else 503,
    }
    assert len(_rows(destination)) == 1
    assert "PRIVATE" not in destination.read_text(encoding="utf-8")
    assert recorder.result.metadata["requests_without_response"] == 1
    assert len(recorder.result.usage) == 1


async def test_concurrent_stages_capture_their_own_failed_response(tmp_path: Path) -> None:
    destination = tmp_path / "failed.jsonl"

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        stage = recorder.stage.get()
        await asyncio.sleep(0)
        return _response({"answers": stage})

    recorder = RecordingModel(FunctionModel(respond), MethodResult(), failure_path=destination)

    async def worker(stage: str) -> None:
        token = recorder.stage.set(stage)
        try:
            await recorder.request(MESSAGES, None, PARAMETERS)
            await asyncio.sleep(0)
            recorder.capture_failure("validation")
        finally:
            recorder.stage.reset(token)

    await asyncio.gather(worker("reduce"), worker("citation_repair"))
    rows = _rows(destination)
    assert {row["stage"] for row in rows} == {"reduce", "citation_repair"}
    for row in rows:
        assert json.loads(row["tools"][0]["arguments"])["answers"] == row["stage"]


@pytest.mark.parametrize("name", list(METHODS))
@pytest.mark.parametrize("problem", ["schema", "question_ids", "citation_invalid"])
async def test_recovered_method_failures_are_saved_but_successes_are_not(
    tmp_path: Path, name: str, problem: str
) -> None:
    calls = 0

    def respond(_messages: Any, info: Any) -> ModelResponse:
        nonlocal calls
        calls += 1
        answer = {
            "question_id": "wrong" if calls == 1 and problem == "question_ids" else "q",
            "source_ids": ["fork:0"],
            "spans": [
                {
                    "quote": "fabricated" if calls == 1 and problem == "citation_invalid" else "Contract text.",
                    "block_id": "b1",
                }
            ],
        }
        payload = {"answers": "invalid"} if calls == 1 and problem == "schema" else {"answers": [answer]}
        output = next(tool for tool in info.output_tools if tool.name == "Answers")
        return ModelResponse(parts=[ToolCallPart(output.name, payload)], finish_reason="stop")

    destination = tmp_path / "failed.jsonl"
    questions = (Question("q", "Which terms apply?"),)
    contract = Contract("synthetic", "Contract text.", questions)
    config = MethodConfig(raw_response_path=str(destination), retries=1, repair_rounds=1)
    result = await METHODS[name](contract, questions, FunctionModel(respond), config)
    assert result.failures == []
    rows = _rows(destination)
    assert len(rows) == 1
    assert rows[0]["classification"] == ("schema_validation" if problem == "schema" else problem)
    assert rows[0]["stage"] == "extraction"
    assert rows[0]["finish_reason"] == "stop"
    assert len(result.usage) == calls
    assert calls > len(rows)
    assert all(row["finish_reason"] == "stop" for row in result.usage)


async def test_reduce_provenance_failure_uses_reduce_response_and_stage(tmp_path: Path) -> None:
    calls = 0

    def respond(_messages: Any, info: Any) -> ModelResponse:
        nonlocal calls
        calls += 1
        payload = {"answers": [{"question_id": "q", "spans": [], "source_ids": ["wrong" if calls == 2 else "fork:0"]}]}
        output = next(tool for tool in info.output_tools if tool.name == "Answers")
        return ModelResponse(parts=[ToolCallPart(output.name, payload)])

    destination = tmp_path / "failed.jsonl"
    questions = (Question("q", "Which terms apply?"),)
    result = await METHODS["forkreduce"](
        Contract("synthetic", "Contract text.", questions),
        questions,
        FunctionModel(respond),
        MethodConfig(raw_response_path=str(destination), retries=1),
    )
    assert result.failures == []
    rows = _rows(destination)
    assert len(rows) == 1
    assert rows[0]["stage"] == "reduce"
    assert rows[0]["failure_reason"] == "reduce_provenance"
    assert json.loads(rows[0]["tools"][0]["arguments"])["answers"][0]["source_ids"] == ["wrong"]
