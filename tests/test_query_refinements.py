"""Literal query layout controls and worker-local contextual output validation."""

from __future__ import annotations

import asyncio
import inspect
import json
import warnings
from dataclasses import asdict
from typing import Any

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.models.test import TestModel
from tests.test_document_layout import (
    LEGACY_DOCUMENTS,
    LEGACY_PROMPTED_TAIL,
    LEGACY_TAIL,
    LEGACY_TOOLS,
    LegacyAnswer,
    user_parts,
)
from tests.test_output_feedback import Answer, Note, RecordingModel, feedback_text
from tests.test_output_specs import DynamicModel, object_spec

from forkreduce import (
    QUERY_SEPARATOR,
    FailureKind,
    ForkReduceError,
    OutputValidationContext,
    Policy,
    PrefixMismatchWarning,
    Query,
    build,
)
from forkreduce.context import estimate_tokens, history_text
from forkreduce.prefix import history_matches_prefix, prefix_hash, prefix_messages, warn_if_prefix_mismatch


@pytest.mark.parametrize(
    ("mode", "digest", "tail"),
    [
        ("tools", "f960f7236503b35896fe25738611cc76a7d1fd9608bbd353ee1833dc02a1f612", LEGACY_TAIL),
        ("prompted", "4c8456c16a9114e986cfba11720f4b9667981e5c3b4ad90dd3f4115eee929f08", LEGACY_PROMPTED_TAIL),
    ],
)
def test_explicit_refinement_defaults_retain_historical_bytes(mode: str, digest: str, tail: bytes) -> None:
    session = build(
        TestModel(),
        documents="alpha\x00beta",
        instructions="Synthetic fixture instructions.",
        outputs=[LegacyAnswer],
        output_mode=mode,  # type: ignore[arg-type]
        query_separator=QUERY_SEPARATOR,
        dynamic_output_instructions=True,
    )
    query = Query(tail="Question.", output=LegacyAnswer)
    assert session.rendered_documents.encode() == LEGACY_DOCUMENTS
    assert session.render_tail(query).encode() == tail
    assert session.prefix_hash == digest
    assert session.tool_specs == LEGACY_TOOLS
    assert [part.content.encode() for part in user_parts(session.history_for(query))] == [LEGACY_DOCUMENTS]


@pytest.mark.parametrize("separator", ["", " ", " \t\r\n ", "\x00\x07\r\n雪"])
@pytest.mark.parametrize("mode", ["tools", "prompted"])
async def test_literal_raw_single_tail_feedback_and_continuation(separator: str, mode: str) -> None:
    recorded = DynamicModel(lambda n: {"value": "invalid" if n == 1 else 7})
    session = build(
        recorded.model,
        documents=" \x00evidence\x07\r\n\t ",
        instructions="Synthetic instructions.",
        dynamic_outputs="generic_tool" if mode == "tools" else "prompted",
        output_mode=mode,  # type: ignore[arg-type]
        document_format="raw",
        message_layout="single",
        query_separator=separator,
        dynamic_output_instructions=False,
        policy=Policy(retries=0, output_feedback_retries=1),
    )
    spec = object_spec({"type": "integer"})
    query = Query(tail=" \tquestion\x00\x07\r\n ", output=spec)
    head = " \x00evidence\x07\r\n\t " + separator + " \tquestion\x00\x07\r\n "
    assert session.render_tail(query) == " \tquestion\x00\x07\r\n "
    assert [part.content for part in user_parts(session.history_for(query))] == [head]
    tools = json.dumps(session.tool_specs, sort_keys=True, default=str)
    assert session.estimate_tokens(query) == estimate_tokens(f"Synthetic instructions.\n{tools}\n{head}")

    first = await session.ask(query)
    assert first.output == {"value": 7}
    assert first.failure is None and first.feedback_retries_used == 1
    assert "integer" in feedback_text(recorded.calls[1][0])
    assert all(user_parts(messages)[0].content == head for messages, _info in recorded.calls)
    assert session.starts_with_prefix(first.messages)

    follow_up = Query(tail=" \r\ncontinue\x00\t ", output=spec, history=first.messages)
    continuation_text = history_text(first.messages)
    assert session.estimate_tokens(follow_up) == estimate_tokens(
        f"Synthetic instructions.\n{tools}\n{continuation_text}\n \r\ncontinue\x00\t "
    )
    second = await session.ask(follow_up)
    assert second.failure is None
    continuation_parts = [part.content for part in user_parts(recorded.calls[-1][0])]
    assert continuation_parts[0] == head
    assert continuation_parts[-1] == " \r\ncontinue\x00\t "
    assert continuation_parts.count(head) == 1
    assert session.starts_with_prefix(second.messages)

    # Single-history validation uses the configured separator literally.
    unrelated = prefix_messages(" \x00evidence\x07\r\n\t ", message_layout="single", tail="different")
    if separator:
        assert not session.starts_with_prefix(unrelated)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.prepare_query(Query(tail="More.", output=spec, history=first.messages))
    assert not any(issubclass(item.category, PrefixMismatchWarning) for item in caught)


@pytest.mark.parametrize("layout", ["split", "single"])
@pytest.mark.parametrize("tail", ["", " ", " \t\r\n ", "\x00\x07雪\r\n\t "])
def test_dynamic_instruction_suppression_is_tail_only(layout: str, tail: str) -> None:
    kwargs: dict[str, Any] = {
        "documents": "Synthetic evidence.",
        "instructions": "Synthetic instructions.",
        "outputs": [Answer],
        "dynamic_outputs": "generic_tool",
        "output_mode": "tools",
        "message_layout": layout,
    }
    default = build(TestModel(), **kwargs)
    literal = build(TestModel(), dynamic_output_instructions=False, **kwargs)
    spec = object_spec({"type": "integer"})
    assert literal.render_tail(Query(tail=tail, output=spec)) == tail
    assert "Use the" in default.render_tail(Query(tail=tail, output=spec))
    fixed = Query(tail=tail, output=Answer)
    assert literal.render_tail(fixed) == default.render_tail(fixed)
    assert literal.prefix_hash == default.prefix_hash
    assert literal.tool_specs == default.tool_specs


@pytest.mark.parametrize("separator", [None, 0, False, b"separator", []])
def test_separator_requires_a_string(separator: Any) -> None:
    with pytest.raises(ForkReduceError, match="query_separator must be a string"):
        build(TestModel(), documents="Evidence.", instructions="", outputs=[Answer], query_separator=separator)
    for operation in (
        lambda: prefix_hash(
            output_mode="tools", instructions="", rendered_documents="x", tools=[], query_separator=separator
        ),
        lambda: prefix_messages("x", query_separator=separator),
        lambda: history_matches_prefix([], "x", query_separator=separator),
    ):
        with pytest.raises(ForkReduceError, match="query_separator must be a string"):
            operation()


@pytest.mark.parametrize("value", [None, 0, 1, "false", []])
def test_dynamic_instruction_switch_requires_bool(value: Any) -> None:
    with pytest.raises(ForkReduceError, match="dynamic_output_instructions must be a bool"):
        build(TestModel(), documents="Evidence.", instructions="", outputs=[Answer], dynamic_output_instructions=value)


def test_custom_separator_hashes_and_prefix_warning_use_literal_contract() -> None:
    kwargs: dict[str, Any] = {"output_mode": "tools", "instructions": "", "rendered_documents": "raw", "tools": []}
    historical = prefix_hash(**kwargs)
    assert prefix_hash(query_separator=QUERY_SEPARATOR, **kwargs) == historical
    assert prefix_hash(query_separator="", **kwargs) != historical
    assert prefix_hash(query_separator=" ", **kwargs) != prefix_hash(query_separator="\t", **kwargs)
    messages = prefix_messages("raw", message_layout="single", query_separator="\x00\t", tail="question")
    assert history_matches_prefix(messages, "raw", message_layout="single", query_separator="\x00\t")
    with pytest.warns(PrefixMismatchWarning):
        warn_if_prefix_mismatch(messages, "raw", prefix_hash_value="synthetic", message_layout="single")


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_contextual_validator_identity_copy_and_native_retry(
    mode: str, dynamic: bool, asynchronous: bool
) -> None:
    recorded = (
        DynamicModel(lambda n: {"value": 0 if n == 1 else 7})
        if dynamic
        else RecordingModel(lambda n: {"value": 0 if n == 1 else 7})
    )
    seen: list[OutputValidationContext] = []

    def validate(output: Any, context: OutputValidationContext) -> Any:
        seen.append(context)
        assert context.query_id == "contextual" and context.round == 4
        assert context.meta["nested"]["label"] == "caller"
        value = output["value"] if dynamic else output.value
        if value == 0:
            raise ModelRetry("Synthetic contextual feedback")
        context.meta["nested"]["label"] = "validator"
        return output

    async def validate_async(output: Any, context: OutputValidationContext) -> Any:
        await asyncio.sleep(0)
        return validate(output, context)

    metadata = {"nested": {"label": "caller"}}
    session = build(
        recorded.model,
        documents="Evidence.",
        instructions="Instructions.",
        outputs=[Answer, Note],
        output_mode=mode,  # type: ignore[arg-type]
        dynamic_outputs=("generic_tool" if mode == "tools" else "prompted") if dynamic else None,
        output_validator=validate_async if asynchronous else validate,
        policy=Policy(retries=0, output_feedback_retries=1),
    )
    output = object_spec({"type": "integer"}) if dynamic else Answer
    result = await session.ask(Query(tail="Question.", output=output, id="contextual", round=4, meta=metadata))
    assert result.failure is None
    assert result.output == ({"value": 7} if dynamic else Answer(value=7))
    assert result.query_id == "contextual" and result.round == 4
    assert result.feedback_retries_used == 1
    assert len(seen) == 2 and seen[0] is seen[1]
    assert metadata == {"nested": {"label": "caller"}}
    assert "Synthetic contextual feedback" in feedback_text(recorded.calls[1][0])
    assert result.meta == {"feedback_retries_used": 1}
    assert session.starts_with_prefix(result.messages)
    if mode == "tools":
        assert [tool.name for tool in recorded.calls[0][1].output_tools][:2] == ["Answer", "Note"]


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("dynamic", [False, True])
async def test_concurrent_contexts_are_query_local(mode: str, dynamic: bool) -> None:
    recorded = DynamicModel(lambda _n: {"value": 7}) if dynamic else RecordingModel(lambda _n: {"value": 7})
    arrived = asyncio.Event()
    seen: dict[str, OutputValidationContext] = {}

    async def validate(output: Any, context: OutputValidationContext) -> Any:
        seen[context.query_id] = context
        if len(seen) == 2:
            arrived.set()
        await asyncio.wait_for(arrived.wait(), timeout=5)
        assert context.meta == {"nested": {"label": context.query_id}}
        assert context.round == (2 if context.query_id == "left" else 5)
        context.meta["nested"]["label"] = "changed"
        return output

    session = build(
        recorded.model,
        documents="Evidence.",
        instructions="Instructions.",
        outputs=[Answer, Note],
        output_mode=mode,  # type: ignore[arg-type]
        dynamic_outputs=("generic_tool" if mode == "tools" else "prompted") if dynamic else None,
        output_validator=validate,
        warmup=False,
        policy=Policy(retries=0),
    )
    output = object_spec({"type": "integer"}) if dynamic else Answer
    left_meta, right_meta = {"nested": {"label": "left"}}, {"nested": {"label": "right"}}
    results = await session.fork(
        [
            Query(tail="Left.", output=output, id="left", round=2, meta=left_meta),
            Query(tail="Right.", output=output, id="right", round=5, meta=right_meta),
        ]
    )
    assert all(result.failure is None for result in results)
    assert seen["left"] is not seen["right"]
    assert seen["left"].meta is not seen["right"].meta
    assert left_meta == {"nested": {"label": "left"}} and right_meta == {"nested": {"label": "right"}}


async def test_context_tracks_assigned_ids_and_continuation_rounds() -> None:
    recorded = RecordingModel(lambda _n: {"value": 7})
    seen: list[dict[str, Any]] = []

    def validate(output: Any, context: OutputValidationContext) -> Any:
        seen.append(asdict(context))
        return output

    session = build(
        recorded.model,
        documents="Evidence.",
        instructions="",
        outputs=[Answer],
        output_validator=validate,
        policy=Policy(retries=0),
    )
    first = await session.ask(Query(tail="First.", output=Answer, round=1, meta={"scope": "first"}))
    second = await session.ask(
        Query(tail="Next.", output=Answer, history=first.messages, round=2, meta={"scope": "next"})
    )
    assert first.failure is second.failure is None
    assert seen == [
        {"query_id": "q1", "round": 1, "meta": {"scope": "first"}},
        {"query_id": "q2", "round": 2, "meta": {"scope": "next"}},
    ]


async def test_contextual_arity_is_inspected_once_and_body_typeerror_is_not_retried() -> None:
    class Validator:
        inspections = 0
        calls = 0

        @property
        def __signature__(self) -> inspect.Signature:
            self.inspections += 1
            return inspect.Signature(
                [
                    inspect.Parameter("output", inspect.Parameter.POSITIONAL_ONLY),
                    inspect.Parameter("context", inspect.Parameter.POSITIONAL_ONLY),
                ]
            )

        def __call__(self, output: Any, context: OutputValidationContext) -> Any:
            self.calls += 1
            raise TypeError("Synthetic callback body TypeError")

    validator = Validator()
    recorded = RecordingModel(lambda _n: {"value": 7})
    session = build(
        recorded.model,
        documents="Evidence.",
        instructions="",
        outputs=[Answer],
        output_validator=validator,
        policy=Policy(retries=0, output_feedback_retries=2),
    )
    results = await session.fork([Query(tail="One.", output=Answer), Query(tail="Two.", output=Answer)])
    assert validator.inspections == 1 and validator.calls == 2
    assert len(recorded.calls) == 2
    assert all(
        result.failure is not None and "Synthetic callback body TypeError" in result.failure.message
        for result in results
    )
    assert all(result.feedback_retries_used == 0 for result in results)


async def test_uninspectable_output_only_validator_preserves_one_argument_behavior() -> None:
    class Validator:
        @property
        def __signature__(self) -> inspect.Signature:
            raise ValueError("Synthetic uninspectable callable")

        def __call__(self, output: Any) -> Any:
            return output

    recorded = RecordingModel(lambda _n: {"value": 7})
    session = build(
        recorded.model,
        documents="Evidence.",
        instructions="",
        outputs=[Answer],
        output_validator=Validator(),
        policy=Policy(retries=0),
    )
    result = await session.ask(Query(tail="Question.", output=Answer))
    assert result.failure is None and result.output == Answer(value=7)


@pytest.mark.parametrize("mode", ["tools", "prompted"])
async def test_contextual_dynamic_validator_does_not_weaken_schema_types(mode: str) -> None:
    recorded = DynamicModel(lambda _n: {"value": "7"})
    seen: list[OutputValidationContext] = []

    def validate(output: Any, context: OutputValidationContext) -> Any:
        seen.append(context)
        return output

    session = build(
        recorded.model,
        documents="Evidence.",
        instructions="",
        output_mode=mode,  # type: ignore[arg-type]
        dynamic_outputs="generic_tool" if mode == "tools" else "prompted",
        dynamic_output_instructions=False,
        output_validator=validate,
        policy=Policy(retries=0),
    )
    result = await session.ask(Query(tail="Literal tail.", output=object_spec({"type": "integer"})))
    assert result.output is None and result.failure is not None
    assert result.failure.kind is FailureKind.VALIDATION
    assert seen == []


async def test_optional_explicit_context_and_variadic_output_only_callbacks() -> None:
    recorded = RecordingModel(lambda _n: {"value": 7})
    contexts: list[OutputValidationContext | None] = []
    argument_counts: list[int] = []

    def contextual(output: Any, context: OutputValidationContext | None = None) -> Any:
        contexts.append(context)
        return output

    def output_only(output: Any, *extra: Any) -> Any:
        argument_counts.append(1 + len(extra))
        return output

    for validator in (contextual, output_only):
        session = build(
            recorded.model,
            documents="Evidence.",
            instructions="",
            outputs=[Answer],
            output_validator=validator,
            policy=Policy(retries=0),
        )
        result = await session.ask(Query(tail="Question.", output=Answer, id="arity"))
        assert result.failure is None and result.output == Answer(value=7)
    assert contexts == [OutputValidationContext(query_id="arity", round=0, meta={})]
    assert argument_counts == [1]
