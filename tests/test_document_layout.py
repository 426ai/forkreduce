"""Offline layout, raw-document, and literal v0.1.1 compatibility fixtures."""

from __future__ import annotations

import copy
import json
from typing import Any, get_args

import pytest
from pydantic import BaseModel
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models.test import TestModel
from tests.support import Scripted, text_result, tool_result
from tests.test_output_feedback import feedback_text

from forkreduce import (
    QUERY_SEPARATOR,
    Block,
    CitationCheck,
    CitationRepair,
    Document,
    DocumentFormat,
    ForkReduceError,
    MessageLayout,
    OutputSpec,
    Policy,
    PrefixMismatchWarning,
    Query,
    build,
)
from forkreduce.context import estimate_tokens, history_text
from forkreduce.documents import coerce_documents, render_documents
from forkreduce.output_specs import DYNAMIC_OUTPUT_TOOL_NAME
from forkreduce.prefix import history_matches_prefix, prefix_hash, prefix_messages, warn_if_prefix_mismatch


class LegacyAnswer(BaseModel):
    """Synthetic fixture answer."""

    value: int


# Captured by executing the original v0.1.1 Session before the layout changes.
# These bytes and digests are literals, not regenerated from current code or git.
LEGACY_DOCUMENTS = b"[document id=doc]\n[block b1]\nalphabeta\n\n===END_FIXED_DOCUMENTS==="
LEGACY_TAIL = b"Question.\n\nUse the `LegacyAnswer` output and no other output type."
LEGACY_PROMPTED_TAIL = (
    LEGACY_TAIL
    + b"\nRespond with a single JSON object matching this schema:\n"
    + b'{"description": "Synthetic fixture answer.", "properties": {"value": {"title": "Value", '
    + b'"type": "integer"}}, "required": ["value"], "title": "LegacyAnswer", "type": "object"}'
)
LEGACY_TOOLS = [
    {
        "name": "LegacyAnswer",
        "description": "Synthetic fixture answer.",
        "schema": {
            "description": "Synthetic fixture answer.",
            "properties": {"value": {"title": "Value", "type": "integer"}},
            "required": ["value"],
            "title": "LegacyAnswer",
            "type": "object",
        },
    }
]


def user_parts(messages: list[ModelMessage]) -> list[UserPromptPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def recording_model(payloads: list[dict[str, Any]], *, output_name: str = "LegacyAnswer"):
    snapshots: list[tuple[list[ModelMessage], Any]] = []

    def respond(messages, info, n):  # type: ignore[no-untyped-def]
        snapshots.append((copy.deepcopy(messages), copy.deepcopy(info)))
        payload = payloads[min(n - 1, len(payloads) - 1)]
        return tool_result(info, output_name, payload) if info.output_tools else text_result(payload)

    return Scripted(respond), snapshots


def test_public_layout_aliases_and_separator() -> None:
    assert get_args(DocumentFormat) == ("blocks", "raw")
    assert get_args(MessageLayout) == ("split", "single")
    assert QUERY_SEPARATOR == "\n\n===QUERY===\n\n"


@pytest.mark.parametrize(
    ("mode", "digest", "tail"),
    [
        ("tools", "f960f7236503b35896fe25738611cc76a7d1fd9608bbd353ee1833dc02a1f612", LEGACY_TAIL),
        ("prompted", "4c8456c16a9114e986cfba11720f4b9667981e5c3b4ad90dd3f4115eee929f08", LEGACY_PROMPTED_TAIL),
    ],
)
def test_literal_v011_bytes_tools_and_hash(mode: str, digest: str, tail: bytes) -> None:
    kwargs: dict[str, Any] = {
        "documents": "alpha\x00beta",
        "instructions": "Synthetic fixture instructions.",
        "outputs": [LegacyAnswer],
        "output_mode": mode,
    }
    session = build(TestModel(), **kwargs)
    explicit = build(TestModel(), document_format="blocks", message_layout="split", **kwargs)
    query = Query(tail="Question.", output=LegacyAnswer)
    assert session.rendered_documents.encode() == explicit.rendered_documents.encode() == LEGACY_DOCUMENTS
    assert session.render_tail(query).encode() == tail
    assert session.tool_specs == explicit.tool_specs == LEGACY_TOOLS
    assert session.prefix_hash == explicit.prefix_hash == digest
    assert session.document_format == session.documents.document_format == "blocks"
    assert session.message_layout == "split"
    assert [part.content.encode() for part in user_parts(session.history_for(query))] == [LEGACY_DOCUMENTS]
    if mode == "tools":
        assert (
            build(TestModel(), **{key: value for key, value in kwargs.items() if key != "output_mode"}).prefix_hash
            == digest
        )


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.asyncio
async def test_default_initial_request_retains_literal_split_parts(mode: str) -> None:
    scripted, snapshots = recording_model([{"value": 3}])
    session = build(
        scripted.model(),
        documents="alpha\x00beta",
        instructions="Synthetic fixture instructions.",
        outputs=[LegacyAnswer],
        output_mode=mode,  # type: ignore[arg-type]
    )
    result = await session.ask(Query(tail="Question.", output=LegacyAnswer))
    assert result.output == LegacyAnswer(value=3)
    messages, info = snapshots[0]
    assert [part.content.encode() for part in user_parts(messages)] == [
        LEGACY_DOCUMENTS,
        LEGACY_TAIL if mode == "tools" else LEGACY_PROMPTED_TAIL,
    ]
    assert info.instructions == "Synthetic fixture instructions."
    assert [tool.name for tool in info.output_tools] == (["LegacyAnswer"] if mode == "tools" else [])


@pytest.mark.parametrize("text", [" α\x00β\x07\r\n\t雪 😀 \n", " \t\r\n ", "\x00", "===END_FIXED_DOCUMENTS==="])
def test_raw_string_is_exact_and_unwrapped(text: str) -> None:
    index = coerce_documents(text, document_format="raw")
    assert index.document_format == "raw"
    assert index.rendered.encode("utf-8") == text.encode("utf-8")
    assert index.documents[0].blocks[0].text == text
    assert index.blocks == {}
    assert render_documents(index.documents, document_format="raw") == text


def test_raw_mixed_order_unused_ids_and_defensive_copies() -> None:
    loose = Block(block_id="b1", text=" loose\x00 ", meta={"source": "private"})
    document = Document(
        doc_id="named",
        title="omit title\x07",
        blocks=[Block(block_id="b1", text="first\r\n"), Block(block_id="", text="\tsecond")],
    )
    index = coerce_documents(["start", loose, document, "finish", "last"], document_format="raw")
    assert index.rendered == "start\n\n loose\x00 \n\nfirst\r\n\n\n\tsecond\n\nfinish\n\nlast"
    assert [block.text for doc in index.documents for block in doc.blocks] == [
        "start",
        " loose\x00 ",
        "first\r\n",
        "\tsecond",
        "finish",
        "last",
    ]
    assert "omit title" not in index.rendered and "private" not in index.rendered
    assert index.documents[1].blocks[0] is not loose
    assert index.documents[2] is not document
    loose.text = "changed"
    loose.meta["source"] = "changed"
    document.blocks[0].text = "changed"
    assert index.documents[1].blocks[0].meta == {"source": "private"}
    assert index.documents[1].blocks[0].text == " loose\x00 "
    assert index.documents[2].blocks[0].text == "first\r\n"


@pytest.mark.parametrize(
    ("documents", "expected"),
    [
        (["one", "two"], "one\n\ntwo"),
        (Block(block_id="", text="one"), "one"),
        ([Block(block_id="same", text="one"), Block(block_id="same", text="two")], "one\n\ntwo"),
        (Document(doc_id="x", blocks=[Block(block_id="x", text="one")]), "one"),
        (
            [
                Document(doc_id="x", blocks=[Block(block_id="same", text="one")]),
                Document(doc_id="x", blocks=[Block(block_id="same", text="two")]),
            ],
            "one\n\ntwo",
        ),
    ],
)
def test_raw_supported_shapes(documents, expected: str) -> None:  # type: ignore[no-untyped-def]
    assert coerce_documents(documents, document_format="raw").rendered == expected


@pytest.mark.parametrize("documents", ["", [], ["", ""], Document(doc_id="empty", blocks=[])])
def test_raw_requires_nonempty_text(documents) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ForkReduceError):
        coerce_documents(documents, document_format="raw")


@pytest.mark.parametrize(
    "options", [{"meta_keys": []}, {"meta_keys": ["source"]}, {"meta_renderer": lambda _d, _b: "meta"}]
)
def test_raw_rejects_metadata_rendering(options: dict[str, Any]) -> None:
    with pytest.raises(ForkReduceError, match="raw.*meta"):
        coerce_documents("one", document_format="raw", **options)
    with pytest.raises(ForkReduceError, match="raw.*meta"):
        render_documents(
            [Document(doc_id="x", blocks=[Block(block_id="x", text="one")])], document_format="raw", **options
        )
    with pytest.raises(ForkReduceError, match="raw.*meta"):
        build(
            TestModel(), documents="one", instructions="rules", outputs=[LegacyAnswer], document_format="raw", **options
        )


@pytest.mark.parametrize("options", [{"document_format": "invalid"}, {"message_layout": "invalid"}])
def test_invalid_layout_options_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ForkReduceError, match="document_format|message_layout"):
        build(TestModel(), documents="one", instructions="rules", outputs=[LegacyAnswer], **options)
    with pytest.raises(ForkReduceError, match="document_format|message_layout"):
        prefix_hash(output_mode="tools", instructions="rules", rendered_documents="one", tools=[], **options)


def test_hash_nondefaults_matter_even_for_identical_text() -> None:
    kwargs = dict(output_mode="tools", instructions="rules", rendered_documents="same bytes", tools=[])
    legacy = prefix_hash(**kwargs)
    assert prefix_hash(**kwargs, document_format="blocks", message_layout="split") == legacy
    assert (
        len(
            {
                legacy,
                prefix_hash(**kwargs, document_format="raw"),
                prefix_hash(**kwargs, message_layout="single"),
                prefix_hash(**kwargs, document_format="raw", message_layout="single"),
            }
        )
        == 4
    )
    clean = build(TestModel(), documents="alpha", instructions="rules", outputs=[LegacyAnswer], document_format="raw")
    control = build(
        TestModel(), documents="al\x00pha", instructions="rules", outputs=[LegacyAnswer], document_format="raw"
    )
    assert clean.prefix_hash != control.prefix_hash


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("document_format", ["blocks", "raw"])
@pytest.mark.parametrize("layout", ["split", "single"])
@pytest.mark.asyncio
async def test_initial_parts_shared_head_and_continuation(mode: str, document_format: str, layout: str) -> None:
    scripted, snapshots = recording_model([{"value": 4}])
    session = build(
        scripted.model(),
        documents=" shared α\x00β \n",
        instructions="Synthetic rules.",
        outputs=[LegacyAnswer],
        output_mode=mode,
        document_format=document_format,
        message_layout=layout,  # type: ignore[arg-type]
        warmup=False,
        policy=Policy(retries=0),
    )
    first_query = Query(tail="First?", output=LegacyAnswer)
    second_query = Query(tail="Second?", output=LegacyAnswer)
    first = await session.ask(first_query)
    second = await session.ask(second_query)
    assert first.output == second.output == LegacyAnswer(value=4)
    first_parts = user_parts(snapshots[0][0])
    second_parts = user_parts(snapshots[1][0])
    if layout == "single":
        head = session.rendered_documents + QUERY_SEPARATOR
        assert [part.content for part in first_parts] == [head + session.render_tail(first_query)]
        assert [part.content for part in second_parts] == [head + session.render_tail(second_query)]
    else:
        assert [part.content for part in first_parts] == [session.rendered_documents, session.render_tail(first_query)]
        assert [part.content for part in second_parts] == [
            session.rendered_documents,
            session.render_tail(second_query),
        ]
    continuation = Query(tail="Continue?", output=LegacyAnswer, history=first.messages)
    original = copy.deepcopy(first.messages)
    continued = await session.ask(continuation)
    assert continued.output == LegacyAnswer(value=4)
    assert first.messages == original
    continued_parts = user_parts(snapshots[2][0])
    assert [part.content for part in continued_parts[:-1]] == [part.content for part in first_parts]
    assert continued_parts[-1].content == session.render_tail(continuation)
    assert sum(str(part.content).count(session.rendered_documents) for part in continued_parts) == 1
    assert session.starts_with_prefix(continued.messages)
    assert snapshots[0][1].output_tools == snapshots[1][1].output_tools == snapshots[2][1].output_tools
    assert all(info.instructions == session.instructions for _, info in snapshots)


def positive(output: Any) -> Any:
    value = output["value"] if isinstance(output, dict) else output.value
    if value <= 0:
        raise ModelRetry("positive value required")
    return output


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("feedback_kind", ["schema", "validator"])
@pytest.mark.asyncio
async def test_single_native_feedback_retains_original_combined_head(
    mode: str, dynamic: bool, feedback_kind: str
) -> None:
    invalid: Any = "not-an-integer" if feedback_kind == "schema" else 0
    output_name = DYNAMIC_OUTPUT_TOOL_NAME if dynamic and mode == "tools" else "LegacyAnswer"
    scripted, snapshots = recording_model([{"value": invalid}, {"value": 5}], output_name=output_name)
    spec = OutputSpec(
        "DynamicAnswer", {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}
    )
    session = build(
        scripted.model(),
        documents=" α\x00β \n",
        instructions="Shared feedback rules.",
        outputs=[] if dynamic else [LegacyAnswer],
        output_mode=mode,  # type: ignore[arg-type]
        dynamic_outputs=("generic_tool" if mode == "tools" else "prompted") if dynamic else None,
        document_format="raw",
        message_layout="single",
        output_validator=positive if feedback_kind == "validator" else None,
        policy=Policy(retries=0, output_feedback_retries=1),
    )
    query: Query[Any] = Query(tail="Validate.", output=spec if dynamic else LegacyAnswer)
    result = await session.ask(query)
    assert result.failure is None
    assert result.output == ({"value": 5} if dynamic else LegacyAnswer(value=5))
    assert result.feedback_retries_used == 1
    assert result.usage.requests == len(snapshots) == 2
    expected = session.rendered_documents + QUERY_SEPARATOR + session.render_tail(query)
    assert [part.content for part in user_parts(snapshots[0][0])] == [expected]
    assert user_parts(snapshots[1][0])[0].content == expected
    assert (
        sum(
            part.content.startswith(session.rendered_documents + QUERY_SEPARATOR)
            for part in user_parts(snapshots[1][0])
        )
        == 1
    )
    assert any(isinstance(message, ModelResponse) for message in snapshots[1][0])
    assert ("integer" if feedback_kind == "schema" else "positive") in feedback_text(snapshots[1][0])
    assert snapshots[0][1].output_tools == snapshots[1][1].output_tools
    assert snapshots[0][1].instructions == snapshots[1][1].instructions == session.instructions
    assert session.starts_with_prefix(result.messages)


def test_single_prefix_requires_separator_boundary_and_copies_history() -> None:
    good = prefix_messages("evidence", message_layout="single", tail="Question")
    assert history_matches_prefix(good, "evidence", message_layout="single")
    assert not history_matches_prefix(prefix_messages("evidenceMore"), "evidence", message_layout="single")
    assert not history_matches_prefix(prefix_messages("evidence"), "evidence", message_layout="single")
    # Historical split matching intentionally remains permissive.
    assert history_matches_prefix(prefix_messages("evidenceMore"), "evidence")
    with pytest.raises(ForkReduceError, match="tail"):
        prefix_messages("evidence", message_layout="single")
    with pytest.warns(PrefixMismatchWarning, match="marker====QUERY==="):
        warn_if_prefix_mismatch(
            prefix_messages("evidence"), "evidence", prefix_hash_value="synthetic", message_layout="single"
        )
    session = build(
        TestModel(), documents="evidence", instructions="rules", outputs=[LegacyAnswer], message_layout="single"
    )
    query = Query(tail="Continue", output=LegacyAnswer, history=good)
    copied = session.history_for(query)
    assert copied == good and copied is not good
    user_parts(copied)[0].content = "changed"
    assert user_parts(good)[0].content == "evidence" + QUERY_SEPARATOR + "Question"


@pytest.mark.parametrize("layout", ["split", "single"])
def test_context_estimate_counts_initial_prefix_separator_and_tail_once(layout: str) -> None:
    session = build(
        TestModel(),
        documents=" synthetic α\x00β ",
        instructions="rules",
        outputs=[LegacyAnswer],
        document_format="raw",
        message_layout=layout,  # type: ignore[arg-type]
    )
    query = Query(tail="Question", output=LegacyAnswer)
    tools = json.dumps(session.tool_specs, sort_keys=True, default=str)
    separator = QUERY_SEPARATOR if layout == "single" else "\n"
    text = f"rules\n{tools}\n{session.rendered_documents}{separator}{session.render_tail(query)}"
    assert session.estimate_tokens(query) == estimate_tokens(text)
    history = session.history_for(query)
    continuation = Query(tail="Continue", output=LegacyAnswer, history=history)
    expected = f"rules\n{tools}\n{history_text(history)}\n{session.render_tail(continuation)}"
    assert session.estimate_tokens(continuation) == estimate_tokens(expected)


def test_raw_block_lookup_and_citation_check_rejected_even_without_citations() -> None:
    index = coerce_documents("synthetic evidence", document_format="raw")
    with pytest.raises(ForkReduceError, match="raw.*addressable"):
        index.block_text("b1")
    with pytest.raises(ForkReduceError, match="CitationCheck.*blocks"):
        CitationCheck()(LegacyAnswer(value=1), documents=index)


@pytest.mark.parametrize("repair", [False, True])
def test_raw_citation_plan_rejected_before_model_call(repair: bool) -> None:
    scripted, snapshots = recording_model([{"value": 1}])
    session = build(
        scripted.model(), documents="raw", instructions="rules", outputs=[LegacyAnswer], document_format="raw"
    )
    plan = session.map([Query(tail="Question", output=LegacyAnswer)])
    with pytest.raises(ForkReduceError, match="blocks"):
        if repair:
            plan.check(lambda _output, *, documents: [], repair=CitationRepair())
        else:
            plan.check(CitationCheck())
    assert snapshots == scripted.calls == []
