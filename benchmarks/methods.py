"""Three offline-testable CUAD strategies with identical answer and repair contracts.

No model is constructed or called at import time. Question gold answers are never
rendered, searched, or passed to a model; only question ids and text cross that boundary.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import replace
from typing import Any

from pydantic_ai import Agent, PromptedOutput, ToolOutput, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from benchmarks.budget import BudgetExceeded, BudgetHalted
from benchmarks.providers import RecordingModel
from benchmarks.schema import (
    FRAMEWORK_DEFAULTS,
    INSTRUCTIONS,
    Answers,
    Contract,
    MethodConfig,
    MethodResult,
    Question,
    question_prompt,
)
from forkreduce import CitationCheck, CitationRepair, Query, build
from forkreduce.documents import Block, Document, DocumentIndex, coerce_documents
from forkreduce.failures import Failure, FailureKind, classify_exception
from forkreduce.policy import Policy
from forkreduce.query import QueryResult


class _IdsError(ValueError):
    pass


def _validate_ids(output: Answers, questions: tuple[Question, ...]) -> None:
    expected = [question.id for question in questions]
    actual = [answer.question_id for answer in output.answers]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise _IdsError("question_ids")
    output.answers.sort(key=lambda answer: expected.index(answer.question_id))


def _last_response(messages: list[ModelMessage]) -> ModelResponse | None:
    return next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)


def _code(exc: BaseException) -> tuple[str, bool, bool]:
    if isinstance(exc, (BudgetExceeded, BudgetHalted)):
        return "budget_exceeded", False, False
    if isinstance(exc, UsageLimitExceeded):
        return "question_budget", False, False
    if isinstance(exc, _IdsError):
        return "question_ids", True, True
    failure = classify_exception(exc)
    code = failure.code
    # Only library-owned codes enter output; never exception text or provider bodies.
    allowed = {"validation", "transport", "context_overflow", "user_error", "budget_exceeded"}
    if code not in allowed and not re.fullmatch(r"http_[0-9]{3}", code):
        code = "unexpected"
    return code, failure.retryable, failure.kind == FailureKind.VALIDATION


def chunk_document(contract: Contract, config: MethodConfig) -> Document:
    """Overlapping addressable sections shared by full-context and retrieval methods."""
    if config.chunk_chars <= 0 or not 0 <= config.chunk_overlap < config.chunk_chars:
        raise ValueError("invalid_chunk_config")
    if not contract.text.strip():
        raise ValueError("empty_contract")
    step = config.chunk_chars - config.chunk_overlap
    blocks = []
    for start in range(0, len(contract.text), step):
        text = contract.text[start : start + config.chunk_chars]
        if text.strip():
            blocks.append(Block(block_id=f"b{len(blocks) + 1}", text=text, meta={"start": str(start)}))
        if start + config.chunk_chars >= len(contract.text):
            break
    return Document(doc_id=contract.id, blocks=blocks)


def _settings(config: MethodConfig) -> dict[str, Any]:
    return {"temperature": config.temperature, "max_tokens": config.max_output_tokens}


def _check_config(config: MethodConfig, questions: tuple[Question, ...]) -> None:
    if len({q.id for q in questions}) != len(questions) or not questions:
        raise ValueError("invalid_questions")
    if (
        config.output_mode not in {"tools", "prompted"}
        or config.concurrency < 1
        or config.retries < 0
        or config.repair_rounds < 0
        or min(
            config.search_results,
            config.read_chars,
            config.requests_per_question,
            config.tool_calls_per_question,
            config.max_output_tokens,
        )
        < 1
    ):
        raise ValueError("invalid_config")


class _ForkExecutor:
    """Retain recovered validation failures, which folded framework results omit."""

    def __init__(
        self, result: MethodResult, model: RecordingModel, config: MethodConfig, questions: tuple[Question, ...]
    ) -> None:
        self.result, self.model, self.config, self.questions = result, model, config, questions
        self.gate = asyncio.Semaphore(config.concurrency)
        self.query_questions = {f"fork:{index}": (q,) for index, q in enumerate(questions)}

    async def run_query(self, session: Any, query: Query[Any]) -> QueryResult[Any]:
        async with self.gate:
            base_id = str(query.meta.get("repair_of", query.id)).split(":repair:")[0]
            expected = self.query_questions.get(base_id, self.questions)
            stage = (
                "citation_repair"
                if "repair" in str(query.id)
                else ("reduce" if str(query.id).startswith("reduce-") else "extraction")
            )
            token = self.model.stage.set(stage)
            try:
                for attempt in range(self.config.retries + 1):
                    current = await session.run_once(query)
                    self.result.metadata["resolved_output_mode"] = current.resolved_output_mode
                    if current.failure is None and current.output is not None:
                        try:
                            _validate_ids(current.output, expected)
                            if stage == "reduce":
                                sources = {q.id: f"fork:{index}" for index, q in enumerate(self.questions)}
                                if any(a.source_ids != [sources[a.question_id]] for a in current.output.answers):
                                    raise _IdsError("reduce_provenance")
                        except _IdsError as exc:
                            self.model.capture_failure(str(exc), response=_last_response(current.messages))
                            current.failure = Failure(FailureKind.VALIDATION, "question_ids", True, "question_ids")
                            current.output = None
                        if (
                            current.output is not None
                            and self.config.raw_response_path is not None
                            and CitationCheck()(current.output, documents=session.documents)
                        ):
                            self.model.capture_failure("citation_invalid", response=_last_response(current.messages))
                    problem = current.failure
                    if problem is not None:
                        self.model.capture_failure(
                            _safe_failure(problem),
                            response=(
                                _last_response(current.messages) if problem.kind == FailureKind.VALIDATION else None
                            ),
                        )
                    if problem is not None and problem.kind == FailureKind.VALIDATION:
                        self.result.schema_failures += 1
                    if problem is None or not problem.retryable or attempt == self.config.retries:
                        return replace(current, attempts=attempt + 1)
                    self.result.retries += 1
            finally:
                self.model.stage.reset(token)
        raise RuntimeError("unreachable")


async def _forkreduce(
    contract: Contract,
    questions: tuple[Question, ...],
    model: RecordingModel,
    config: MethodConfig,
    result: MethodResult,
) -> None:
    document = chunk_document(contract, config)
    session = build(
        model,
        documents=document,
        instructions=INSTRUCTIONS,
        outputs=[Answers],
        output_mode=config.output_mode,
        on_unforced_tools="ignore",
        dynamic_outputs=None,
        document_format="blocks",
        message_layout="split",
        instrument=False,
        warmup=True,
        policy=Policy(
            concurrency=config.concurrency, retries=config.retries, retry_backoff=0, output_feedback_retries=0
        ),
        model_settings=_settings(config),
        executor=_ForkExecutor(result, model, config, questions),
    )
    result.metadata.update(prefix_hash=session.prefix_hash, resolved_output_mode=session.resolved_output_mode)
    queries = [
        Query(tail=question_prompt((q,)), output=Answers, id=f"fork:{index}") for index, q in enumerate(questions)
    ]
    run = await (
        session.map(queries)
        .check(CitationCheck(), repair=CitationRepair(max_rounds=config.repair_rounds))
        .reduce(
            output=Answers,
            max_batch=max(200, len(questions)),
            instructions=(
                question_prompt(questions) + "\nPreserve every question_id exactly once, including no-answer entries. "
                "Merge only answers to the same question; never substitute another question's citations. "
                "Keep verbatim quotes and block_ids. Replace, do not union, any inherited source_ids. "
                "For each question_id, source_ids must equal the list in this mapping, even for empty spans: "
                + json.dumps({q.id: [f"fork:{index}"] for index, q in enumerate(questions)})
                + ". Do not put contract/document ids or block_ids in source_ids."
            ),
        )
        .run()
    )
    for failure in run.failures:
        result.failures.append(_safe_failure(failure))
    for report in run.reports:
        if getattr(report, "failures", None):
            result.failures.append("citation_invalid")
    if not isinstance(run.merged, Answers):
        result.failures.append("missing_output")
        return
    result.answers = run.merged
    _validate_ids(result.answers, questions)
    expected_sources = {q.id: f"fork:{index}" for index, q in enumerate(questions)}
    for answer in result.answers.answers:
        if answer.source_ids != [expected_sources[answer.question_id]]:
            result.failures.append("reduce_provenance")
    reduce_results = [r for r in run.results if r.query_id.startswith("reduce-")]
    if not reduce_results:
        result.failures.append("missing_reduce")
        return
    carrier = replace(reduce_results[-1], output=result.answers)
    check, repair = CitationCheck(), CitationRepair(max_rounds=config.repair_rounds)
    problems = check(carrier.output, documents=session.documents)
    for round_index in range(1, config.repair_rounds + 1):
        if not problems:
            break
        repaired = await session.ask(repair.build_query(carrier, problems, round_index=round_index))
        if repaired.failure is not None or repaired.output is None:
            result.failures.append(_safe_failure(repaired.failure) if repaired.failure else "missing_output")
            break
        try:
            updated = repair.apply(carrier.output, repaired.output, problems)
        except Exception:
            result.failures.append("repair_apply")
            break
        carrier = replace(repaired, output=updated)
        problems = check(updated, documents=session.documents)
    result.answers = carrier.output
    if problems:
        result.failures.append("citation_invalid")


def _safe_failure(failure: Failure) -> str:
    code = failure.code
    allowed = {
        "validation",
        "question_ids",
        "wrong_output_type",
        "transport",
        "context_overflow",
        "user_error",
        "budget_exceeded",
        "check_failed",
        "repair_apply",
        "reduce_failed",
        "reduce_did_not_shrink",
    }
    return code if code in allowed or re.fullmatch(r"http_[0-9]{3}", code) else "unexpected"


async def _agent_call(
    agent: Agent[Any, Any],
    prompt: str,
    history: list[ModelMessage],
    questions: tuple[Question, ...],
    model: RecordingModel,
    config: MethodConfig,
    result: MethodResult,
    usage: RunUsage,
    limits: UsageLimits,
    stage: str,
) -> tuple[Answers | None, list[ModelMessage]]:
    token = model.stage.set(stage)
    try:
        for attempt in range(config.retries + 1):
            with capture_run_messages() as captured:
                try:
                    response = await agent.run(prompt, message_history=history, usage=usage, usage_limits=limits)
                    history = list(response.all_messages())
                    _validate_ids(response.output, questions)
                    return response.output, history
                except Exception as exc:
                    history = list(captured) or history
                    code, retryable, schema = _code(exc)
                    model.capture_failure(code, response=_last_response(history) if schema else None)
                    if schema:
                        result.schema_failures += 1
                    if not retryable or attempt == config.retries:
                        result.failures.append(code)
                        return None, history
                    result.retries += 1
                    prompt = (
                        question_prompt(questions) + "\nReturn the complete Answers schema with exact question ids."
                    )
    finally:
        model.stage.reset(token)
    return None, history


async def _agent_answer(
    agent: Agent[Any, Any],
    prompt: str,
    history: list[ModelMessage],
    questions: tuple[Question, ...],
    documents: DocumentIndex,
    model: RecordingModel,
    config: MethodConfig,
    result: MethodResult,
    limits: UsageLimits,
) -> tuple[Answers | None, list[ModelMessage]]:
    usage = RunUsage()
    output, history = await _agent_call(
        agent, prompt, history, questions, model, config, result, usage, limits, "extraction"
    )
    if output is None:
        return None, history
    check, repair = CitationCheck(), CitationRepair(max_rounds=config.repair_rounds)
    problems = check(output, documents=documents)
    if problems:
        model.capture_failure("citation_invalid", response=_last_response(history))
    for round_index in range(1, config.repair_rounds + 1):
        if not problems:
            break
        carrier = QueryResult("agent", output, history, "", usage, 0, 1, None, 0)
        query = repair.build_query(carrier, problems, round_index=round_index)
        updated, history = await _agent_call(
            agent, query.tail, history, questions, model, config, result, usage, limits, "citation_repair"
        )
        if updated is None:
            break
        try:
            output = repair.apply(output, updated, problems)
        except Exception:
            result.failures.append("repair_apply")
            break
        problems = check(output, documents=documents)
        if problems:
            token = model.stage.set("citation_repair")
            try:
                model.capture_failure("citation_invalid", response=_last_response(history))
            finally:
                model.stage.reset(token)
    if problems:
        result.failures.append("citation_invalid")
    return output, history


def _agent(model: Model, config: MethodConfig) -> Agent[Any, Answers]:
    agent: Agent[Any, Answers] = Agent(
        model,
        output_type=(
            ToolOutput(Answers, name="Answers", max_retries=0)
            if config.output_mode == "tools"
            else PromptedOutput(Answers)
        ),
        instructions=INSTRUCTIONS,
        retries=0,
        model_settings=_settings(config),
    )
    agent.instrument = False
    return agent


async def _one_shot(
    contract: Contract,
    questions: tuple[Question, ...],
    model: RecordingModel,
    config: MethodConfig,
    result: MethodResult,
) -> None:
    documents = coerce_documents(chunk_document(contract, config))
    agent = _agent(model, config)
    output, _ = await _agent_answer(
        agent,
        documents.rendered + "\n\n" + question_prompt(questions),
        [],
        questions,
        documents,
        model,
        config,
        result,
        UsageLimits(request_limit=None),
    )
    if output is not None:
        result.answers = output


async def _tool_agent(
    contract: Contract,
    questions: tuple[Question, ...],
    model: RecordingModel,
    config: MethodConfig,
    result: MethodResult,
) -> None:
    documents = coerce_documents(chunk_document(contract, config))
    agent = _agent(model, config)
    calls = {"search": 0, "read_section": 0}
    block_items = list(documents.blocks.items())
    terms_by_block = [set(re.findall(r"\w+", block.text.casefold())) for _, block in block_items]

    @agent.tool_plain
    def search(query: str) -> list[dict[str, Any]]:
        """Search overlapping contract sections. Returns ids, offsets, and bounded excerpts."""
        calls["search"] += 1
        terms = set(re.findall(r"\w+", query.casefold()))
        ranked = sorted(enumerate(terms_by_block), key=lambda pair: (-len(terms & pair[1]), pair[0]))
        return [
            {
                "id": block_items[index][0],
                "start": int(block_items[index][1].meta["start"]),
                "length": len(block_items[index][1].text),
                "excerpt": block_items[index][1].text[:800],
            }
            for index, _ in ranked[: config.search_results]
        ]

    @agent.tool_plain
    def read_section(id: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
        """Read a section by id with section-relative [start,end) character offsets."""
        calls["read_section"] += 1
        block = documents.blocks.get(id)
        if block is None:
            return {"error": "unknown_section"}
        stop = min(len(block.text), start + config.read_chars, end if end is not None else len(block.text))
        if start < 0 or start > len(block.text) or stop < start:
            return {"error": "invalid_range"}
        return {"id": id, "start": start, "end": stop, "text": block.text[start:stop], "length": len(block.text)}

    history: list[ModelMessage] = []
    for question in questions:
        prompt = (
            question_prompt((question,)) + "\nUse search(query) and read_section(id,start,end) to inspect "
            "the contract as needed. Section ids are b1, b2, ... in document order. "
            f"The contract has {len(block_items)} overlapping sections. "
            "Answer only this question; previous turns are retained context, not additional questions."
        )
        output, history = await _agent_answer(
            agent,
            prompt,
            history,
            (question,),
            documents,
            model,
            config,
            result,
            UsageLimits(request_limit=config.requests_per_question, tool_calls_limit=config.tool_calls_per_question),
        )
        if output is not None:
            result.answers.answers.extend(output.answers)
    result.metadata["tool_calls"] = calls


async def _run(
    name: str,
    implementation: Any,
    contract: Contract,
    questions: tuple[Question, ...],
    model: Model,
    config: MethodConfig,
) -> MethodResult:
    result = MethodResult(
        concurrency=config.concurrency if name == "forkreduce" else 1,
        metadata={"method": name, "framework_defaults": dict(FRAMEWORK_DEFAULTS)},
    )
    started = time.perf_counter()
    try:
        questions = tuple(questions)
        _check_config(config, questions)
        result.metadata["requested_output_mode"] = config.output_mode
        if name == "tool_agent" and config.output_mode == "prompted":
            result.failures.append("tool_agent_prompted_unsupported")
            return result
        await implementation(
            contract, questions, RecordingModel(model, result, failure_path=config.raw_response_path), config, result
        )
        _validate_ids(result.answers, questions)
    except Exception as exc:
        result.failures.append(_code(exc)[0])
    finally:
        result.elapsed_seconds = time.perf_counter() - started
    return result


async def forkreduce(
    contract: Contract, questions: tuple[Question, ...], model: Model, config: MethodConfig
) -> MethodResult:
    return await _run("forkreduce", _forkreduce, contract, questions, model, config)


async def one_shot(
    contract: Contract, questions: tuple[Question, ...], model: Model, config: MethodConfig
) -> MethodResult:
    return await _run("one_shot", _one_shot, contract, questions, model, config)


async def tool_agent(
    contract: Contract, questions: tuple[Question, ...], model: Model, config: MethodConfig
) -> MethodResult:
    return await _run("tool_agent", _tool_agent, contract, questions, model, config)


METHODS = {"forkreduce": forkreduce, "tool_agent": tool_agent, "one_shot": one_shot}


def offline_model() -> Model:
    """Deterministic synthetic smoke model, never a substitute for a live benchmark.

    It recognizes renewal and termination sentences and abstains on other questions.
    In retrieval mode it deliberately executes search then read_section every turn.
    """
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RequestUsage

    def respond(messages: list[ModelMessage], info: Any) -> ModelResponse:
        prompts = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ]
        latest = prompts[-1]
        text = "\n".join(prompts)
        usage = RequestUsage(input_tokens=200, output_tokens=40, cache_read_tokens=100)
        last_parts = messages[-1].parts
        available = {tool.name for tool in info.function_tools}
        if "search" in available:
            if any(isinstance(part, UserPromptPart) for part in last_parts):
                return ModelResponse(parts=[ToolCallPart("search", {"query": latest})], usage=usage)
            returns = [part for part in last_parts if isinstance(part, ToolReturnPart)]
            if returns and returns[-1].tool_name == "search":
                return ModelResponse(parts=[ToolCallPart("read_section", {"id": "b1", "start": 0})], usage=usage)
        pairs = re.findall(r"(?m)^(.+?): (.+)$", latest.split("Answer these questions in order:\n")[-1])
        # Repair continuations carry the original question turn in retained history.
        if "Answer these questions in order:" not in latest:
            prior = next((prompt for prompt in reversed(prompts) if "Answer these questions in order:" in prompt), "")
            pairs = re.findall(r"(?m)^(.+?): (.+)$", prior.split("Answer these questions in order:\n")[-1])
        pairs = [(qid, question) for qid, question in pairs if qid not in {"id", "Hint", "Failures"}]
        if "INPUT_ITEMS" in latest:
            payloads = re.findall(r"(?m)^id: (\S+)\n([^\n]+)", latest)
            answers = []
            for source_id, payload in payloads:
                for answer in json.loads(payload)["answers"]:
                    answer["source_ids"] = [source_id]
                    answers.append(answer)
        else:
            corpus = text.split("===END_FIXED_DOCUMENTS===")[0] if "[block " in text else ""
            for message in messages:
                for part in message.parts:
                    if isinstance(part, ToolReturnPart) and part.tool_name == "read_section":
                        content = part.content
                        if isinstance(content, dict):
                            corpus += "\n" + str(content.get("text", ""))
            answers = []
            for qid, question in pairs:
                needle = (
                    "renew"
                    if "renew" in question.casefold()
                    else ("terminat" if "terminat" in question.casefold() else None)
                )
                sentences = re.split(r"(?<=[.!?])\s+|\n", corpus)
                quote = next((s for s in sentences if needle and needle in s.casefold() and not s.startswith("[")), "")
                answers.append(
                    {
                        "question_id": qid,
                        "spans": ([{"quote": quote, "block_id": "b1"}] if quote else []),
                        "source_ids": [],
                    }
                )
        if not info.output_tools:
            return ModelResponse(parts=[TextPart(json.dumps({"answers": answers}))], usage=usage)
        output = next(tool for tool in info.output_tools if tool.name == "Answers")
        return ModelResponse(parts=[ToolCallPart(output.name, {"answers": answers})], usage=usage)

    return FunctionModel(respond)
