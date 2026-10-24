"""Synthetic benchmark method coverage: no dataset, credentials, or network calls."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from typing import Any

import pytest
from benchmarks.methods import METHODS, chunk_document, offline_model
from benchmarks.providers import provider_model, usage_record
from benchmarks.schema import Answers, Contract, MethodConfig, Question, Span
from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

TEXT = "The agreement renews annually.\nEither party may terminate on thirty days notice."
QUESTIONS = (
    Question("renewal", "Which renewal terms apply?", (Span("GOLD_SECRET_DO_NOT_RENDER", 0),)),
    Question("absent", "Is there an exclusivity clause?"),
)
CONTRACT = Contract("synthetic", TEXT, QUESTIONS)
CONFIG = MethodConfig(concurrency=3, retries=1, repair_rounds=2)


def _prompts(messages: Any) -> list[str]:
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


def _reply(info: Any, data: Any) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(next(t.name for t in info.output_tools if t.name == "Answers"), data)],
        usage=RequestUsage(input_tokens=100, output_tokens=20, cache_read_tokens=60, cache_write_tokens=10),
    )


def _scripted(
    *, invalid_schema: bool = False, invalid_citations: bool = False, invalid_reduce: bool = False
) -> tuple[FunctionModel, list[Any]]:
    calls: list[Any] = []

    def respond(messages: Any, info: Any) -> ModelResponse:
        calls.append((messages, info))
        prompts = _prompts(messages)
        assert "GOLD_SECRET" not in "\n".join(prompts)
        latest = prompts[-1]
        if invalid_schema and len(calls) == 1:
            return _reply(info, {"answers": "invalid"})
        if "Some citations could not be located" in latest:
            previous = next(
                part
                for message in reversed(messages)
                for part in reversed(message.parts)
                if isinstance(part, ToolCallPart) and part.tool_name == "Answers"
            )
            payload = previous.args_as_dict()
            for answer in payload["answers"]:
                for citation in answer.get("spans", []):
                    citation["quote"] = "The agreement renews annually."
                    citation["block_id"] = "b1"
                # CitationRepair must not copy arbitrary non-citation fields.
                answer["source_ids"] = ["untrusted_repair_change"]
            return _reply(info, payload)
        if "INPUT_ITEMS" in latest:
            answers = []
            for source, body in re.findall(r"(?m)^id: (\S+)\n([^\n]+)", latest):
                for answer in json.loads(body)["answers"]:
                    answer["source_ids"] = [source]
                    if invalid_reduce:
                        for citation in answer.get("spans", []):
                            citation["quote"] = "fabricated reduced citation"
                    answers.append(answer)
            return _reply(info, {"answers": answers})
        requested = latest.split("Answer these questions in order:\n")[-1]
        ids = re.findall(r"(?m)^(renewal|absent):", requested)
        answers = [
            {
                "question_id": qid,
                "source_ids": [],
                "spans": (
                    [
                        {
                            "quote": "fabricated citation" if invalid_citations else "The agreement renews annually.",
                            "block_id": "b1",
                        }
                    ]
                    if qid == "renewal"
                    else []
                ),
            }
            for qid in ids
        ]
        return _reply(info, {"answers": answers})

    return FunctionModel(respond), calls


@pytest.mark.parametrize("name,requests", [("forkreduce", 3), ("one_shot", 1), ("tool_agent", 6)])
async def test_all_methods_structured_no_answer_and_actual_tools(name: str, requests: int) -> None:
    result = await METHODS[name](CONTRACT, QUESTIONS, offline_model(), CONFIG)
    assert result.failures == []
    assert isinstance(result.answers, Answers)
    assert [answer.question_id for answer in result.answers.answers] == ["renewal", "absent"]
    assert result.answers.answers[0].spans[0].quote == "The agreement renews annually."
    assert result.answers.answers[1].spans == []
    assert len(result.usage) == requests
    assert sum(row["input_tokens"] for row in result.usage) == 200 * requests
    assert sum(row["cache_read_tokens"] for row in result.usage) == 100 * requests
    assert result.elapsed_seconds > 0
    assert result.concurrency == (3 if name == "forkreduce" else 1)
    if name == "tool_agent":
        assert result.metadata["tool_calls"] == {"search": 2, "read_section": 2}
    if name == "forkreduce":
        assert [a.source_ids for a in result.answers.answers] == [["fork:0"], ["fork:1"]]
        assert result.usage[-1]["stage"] == "reduce"


@pytest.mark.parametrize("name", list(METHODS))
async def test_recovered_schema_failure_is_counted_and_billed_once(name: str) -> None:
    model, calls = _scripted(invalid_schema=True)
    result = await METHODS[name](CONTRACT, QUESTIONS, model, CONFIG)
    assert result.failures == []
    assert result.schema_failures == 1
    assert result.retries == 1
    assert len(result.usage) == len(calls)
    assert sum(row["input_tokens"] for row in result.usage) == len(calls) * 100
    assert sum(row["cache_write_tokens"] for row in result.usage) == len(calls) * 10


@pytest.mark.parametrize("name", list(METHODS))
async def test_identical_field_limited_citation_repair(name: str) -> None:
    model, calls = _scripted(invalid_citations=True)
    result = await METHODS[name](CONTRACT, QUESTIONS, model, CONFIG)
    assert result.failures == []
    assert result.answers.answers[0].spans[0].quote == "The agreement renews annually."
    assert result.answers.answers[1].spans == []
    assert any(row["stage"] == "citation_repair" for row in result.usage)
    assert len(result.usage) == len(calls)
    assert all("untrusted_repair_change" not in answer.source_ids for answer in result.answers.answers)


async def test_reduce_citations_are_checked_after_semantic_merge() -> None:
    model, calls = _scripted(invalid_reduce=True)
    result = await METHODS["forkreduce"](CONTRACT, QUESTIONS, model, CONFIG)
    assert result.failures == []
    assert result.answers.answers[0].spans[0].quote == "The agreement renews annually."
    assert result.answers.answers[0].source_ids == ["fork:0"]
    assert result.usage[-1]["stage"] == "citation_repair"
    assert len(result.usage) == len(calls) == 4


@pytest.mark.parametrize("name", list(METHODS))
async def test_testmodel_support_and_exact_id_validation(name: str) -> None:
    questions = (Question("absent", "Is there an exclusivity clause?"),)
    # A wrong/duplicate/extra id may never silently enter reported answers.
    model = TestModel(
        call_tools=[],
        custom_output_args={"answers": [{"question_id": "wrong", "spans": []}, {"question_id": "wrong", "spans": []}]},
    )
    result = await METHODS[name](CONTRACT, questions, model, replace(CONFIG, retries=0))
    assert "question_ids" in result.failures
    assert result.schema_failures == 1
    assert result.retries == 0
    assert len(result.usage) == 1


@pytest.mark.parametrize("name", list(METHODS))
async def test_testmodel_no_answer_remains_valid(name: str) -> None:
    questions = (Question("absent", "Is there an exclusivity clause?"),)
    model = TestModel(
        call_tools=[],
        custom_output_args={"answers": [{"question_id": "absent", "spans": [], "source_ids": ["fork:0"]}]},
    )
    result = await METHODS[name](CONTRACT, questions, model, CONFIG)
    assert result.failures == []
    assert result.answers.answers[0].spans == []


@pytest.mark.parametrize("name", list(METHODS))
async def test_failed_method_is_safe_and_does_not_raise(name: str) -> None:
    def fail(_messages: Any, _info: Any) -> ModelResponse:
        raise RuntimeError("secret API_KEY=DO_NOT_PERSIST")

    result = await METHODS[name](CONTRACT, QUESTIONS, FunctionModel(fail), CONFIG)
    assert result.failures
    assert "secret" not in repr(result)
    assert "API_KEY" not in repr(result)
    assert result.elapsed_seconds > 0
    assert result.metadata["requests_without_response"] > 0


async def test_recorders_are_run_local_under_concurrency() -> None:
    left, right = await asyncio.gather(
        METHODS["one_shot"](CONTRACT, QUESTIONS, offline_model(), CONFIG),
        METHODS["tool_agent"](CONTRACT, QUESTIONS, offline_model(), CONFIG),
    )
    assert len(left.usage) == 1
    assert len(right.usage) == 6
    assert left.usage is not right.usage


def test_chunk_overlap_has_stable_ids_and_exact_source_text() -> None:
    contract = Contract("x", "abcdefghijklmnop", ())
    document = chunk_document(contract, replace(CONFIG, chunk_chars=8, chunk_overlap=3))
    assert [b.block_id for b in document.blocks] == ["b1", "b2", "b3"]
    assert [b.text for b in document.blocks] == ["abcdefgh", "fghijklm", "klmnop"]
    assert [b.meta["start"] for b in document.blocks] == ["0", "5", "10"]


def test_usage_fallback_and_missing_are_not_fabricated() -> None:
    fallback = ModelResponse(
        parts=[],
        usage=RequestUsage(
            input_tokens=100,
            output_tokens=10,
            details={"prompt_cache_hit_tokens": 70, "cache_creation_input_tokens": 20},
        ),
    )
    row = usage_record(fallback, stage="test")
    assert row["cache_read_tokens"] == 70
    assert row["cache_write_tokens"] == 20
    assert row["input_tokens"] == 100
    assert row["usage_missing"] is False
    missing = usage_record(ModelResponse(parts=[]), stage="test")
    assert missing["usage_missing"] is True
    assert missing["input_tokens"] is None


def test_provider_factory_does_not_dump_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-token")
    with pytest.raises(ValueError, match="^anthropic_model_missing$"):
        provider_model("anthropic")


def test_provider_usage_details_only_persist_numeric_allowlist() -> None:
    from types import SimpleNamespace

    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
            details={
                "prompt_cache_hit_tokens": 7,
                "API_KEY": "secret",
                "unexpected_numeric": 42,
                "prompt_tokens_details": {"cached_tokens": 8, "secret": "token"},
            },
        )
    )
    record = usage_record(response, stage="test")
    assert record["details"] == {"prompt_cache_hit_tokens": 7, "cached_tokens": 8}
    assert "secret" not in repr(record)


async def test_tool_budget_is_cumulative_across_retries_and_question_repairs() -> None:
    result = await METHODS["tool_agent"](CONTRACT, QUESTIONS, offline_model(), replace(CONFIG, requests_per_question=2))
    assert "question_budget" in result.failures
    assert result.retries == 0
    assert len(result.usage) <= 4


async def test_fork_warmup_finishes_before_other_questions_start() -> None:
    entered = []
    completed = []

    async def respond(messages: Any, info: Any) -> ModelResponse:
        latest = _prompts(messages)[-1]
        if "INPUT_ITEMS" in latest:
            answers = []
            for source, body in re.findall(r"(?m)^id: (\S+)\n([^\n]+)", latest):
                for answer in json.loads(body)["answers"]:
                    answer["source_ids"] = [source]
                    answers.append(answer)
            return _reply(info, {"answers": answers})
        qid = "renewal" if "renewal:" in latest else "absent"
        entered.append(qid)
        if qid == "absent":
            assert completed == ["renewal"]
        await asyncio.sleep(0)
        completed.append(qid)
        return _reply(info, {"answers": [{"question_id": qid, "spans": []}]})

    result = await METHODS["forkreduce"](CONTRACT, QUESTIONS, FunctionModel(respond), CONFIG)
    assert result.failures == []
    assert entered == ["renewal", "absent"]
