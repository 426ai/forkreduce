"""Prefix identity, documents, and session guards."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from tests.support import Scripted, text_result, tool_result, user_text

import forkreduce
from forkreduce import (
    END_MARKER,
    Block,
    Document,
    PrefixMismatchWarning,
    Query,
    UnregisteredOutputError,
    build,
)
from forkreduce.documents import coerce_documents
from forkreduce.policy import Policy


class Answer(BaseModel):
    """A short grounded answer."""

    text: str
    quote: str
    block_id: str


class Note(BaseModel):
    """A note."""

    body: str


def _scripted(payload: dict[str, object] | None = None) -> Scripted:
    body = payload or {"text": "ok", "quote": "alpha", "block_id": "b1"}

    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        if info.output_tools:
            return tool_result(info, "Answer", dict(body))
        return text_result(dict(body))

    return Scripted(handler)


def test_control_characters_do_not_change_the_prefix_hash() -> None:
    clean = build(
        _scripted().model(),
        documents="hello",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
    )
    dirty = build(
        _scripted().model(),
        documents="hel\x00lo\x07",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
    )
    assert clean.prefix_hash == dirty.prefix_hash
    assert "\x00" not in dirty.rendered_documents
    assert "hello" in dirty.rendered_documents
    assert dirty.rendered_documents.endswith(END_MARKER) or END_MARKER in dirty.rendered_documents


def test_hash_changes_when_documents_or_tools_change() -> None:
    left = build(
        _scripted().model(),
        documents="alpha",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
    )
    right = build(
        _scripted().model(),
        documents="beta",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
    )
    extra = build(
        _scripted().model(),
        documents="alpha",
        instructions="rules",
        outputs=[Answer, Note],
        warmup=False,
    )
    again = build(
        _scripted().model(),
        documents="alpha",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
    )
    assert left.prefix_hash != right.prefix_hash
    assert left.prefix_hash != extra.prefix_hash
    assert left.prefix_hash == again.prefix_hash


def test_duplicate_block_id_is_rejected() -> None:
    with pytest.raises(forkreduce.ForkReduceError, match="duplicate"):
        coerce_documents(
            Document(
                doc_id="doc",
                blocks=[Block(block_id="b1", text="one"), Block(block_id="b1", text="two")],
            )
        )


@pytest.mark.asyncio
async def test_unregistered_output_raises_before_the_model_is_called() -> None:
    scripted = _scripted()
    session = build(
        scripted.model(),
        documents="alpha beta",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )

    class Other(BaseModel):
        value: int

    with pytest.raises(UnregisteredOutputError):
        await session.ask(Query(tail="nope", output=Other))
    assert scripted.calls == []


@pytest.mark.asyncio
async def test_history_continuation_and_prefix_mismatch_warning() -> None:
    scripted = _scripted()
    session = build(
        scripted.model(),
        documents=Block(block_id="b1", text="alpha beta"),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    first = await session.ask(Query(tail="What is alpha?", output=Answer, id="q1"))
    assert session.starts_with_prefix(first.messages)
    assert first.output is not None
    assert first.output.text == "ok"

    second = await session.ask(Query(tail="And then?", output=Answer, id="q2", history=first.messages))
    seen = user_text(scripted.calls[-1][0])
    assert "What is alpha?" in seen
    assert "And then?" in seen
    assert session.starts_with_prefix(second.messages)
    assert len(second.messages) > len(first.messages)

    with pytest.warns(PrefixMismatchWarning):
        await session.ask(
            Query(
                tail="orphan",
                output=Answer,
                id="q3",
                history=[ModelRequest(parts=[UserPromptPart(content="not the prefix")])],
            )
        )


@pytest.mark.asyncio
async def test_tool_set_is_identical_for_every_query() -> None:
    seen: list[tuple[str, ...]] = []
    instructions: list[str] = []

    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        seen.append(tuple(tool.name for tool in info.output_tools))
        instructions.append(info.instructions or "")
        name = "Note" if "note" in user_text(messages) else "Answer"
        payload = {"body": "n"} if name == "Note" else {"text": "ok", "quote": "alpha", "block_id": "b1"}
        return tool_result(info, name, payload)

    session = build(
        Scripted(handler).model(),
        documents="alpha",
        instructions="shared rules",
        outputs=[Answer, Note],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0, concurrency=4),
    )
    results = await session.fork(
        [
            Query(tail="answer this", output=Answer, id="a"),
            Query(tail="make a note", output=Note, id="b"),
        ]
    )
    assert seen == [("Answer", "Note"), ("Answer", "Note")]
    assert instructions == ["shared rules", "shared rules"]
    assert {type(result.output).__name__ for result in results} == {"Answer", "Note"}


@pytest.mark.asyncio
async def test_prompted_mode_puts_schema_in_the_tail_only() -> None:
    captured: dict[str, str] = {}

    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        captured["instructions"] = info.instructions or ""
        captured["user"] = user_text(messages)
        assert info.output_tools == []
        assert info.allow_text_output
        return text_result({"text": "ok", "quote": "alpha", "block_id": "b1"})

    session = build(
        Scripted(handler).model(),
        documents="alpha",
        instructions="rules",
        outputs=[Answer, Note],
        output_mode="prompted",
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    result = await session.ask(Query(tail="Question?", output=Answer, id="p"))
    assert result.output is not None
    assert result.output.text == "ok"
    assert "properties" not in captured["instructions"]
    assert "Question?" in captured["user"]
    assert '"quote"' in captured["user"] or "quote" in captured["user"]


@pytest.mark.asyncio
async def test_testmodel_can_drive_a_single_registered_output() -> None:
    session = build(
        TestModel(custom_output_args={"text": "from-test", "quote": "alpha", "block_id": "b1"}),
        documents="alpha",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    result = await session.ask(Query(tail="ping", output=Answer, id="t"))
    assert result.failure is None
    assert result.output is not None
    assert result.output.text == "from-test"
    assert result.prefix_hash == session.prefix_hash


def test_mixed_document_input_keeps_caller_order() -> None:
    index = coerce_documents(
        [
            Block(block_id="a", text="alpha"),
            Document(doc_id="mid", title="Middle", blocks=[Block(block_id="b", text="beta")]),
            "tail",
        ]
    )
    rendered = index.rendered
    assert rendered.index("[block a]") < rendered.index("id=mid") < rendered.index("tail")
    assert [doc.doc_id for doc in index.documents] == ["doc", "mid", "doc-2"]


def _notice(meta: dict[str, str] | None = None) -> Block:
    return Block(block_id="b1", text="Hours are 9 to 5.", meta=dict(meta or {}))


def test_block_meta_is_omitted_from_the_prefix_by_default() -> None:
    plain = build(
        _scripted().model(),
        documents=_notice(),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
    )
    tagged = build(
        _scripted().model(),
        documents=_notice({"page": "3", "section": "PAGE-TOKEN"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
    )
    assert plain.prefix_hash == tagged.prefix_hash
    assert "PAGE-TOKEN" not in tagged.rendered_documents
    assert "[meta " not in tagged.rendered_documents
    assert tagged.documents.blocks["b1"].meta["section"] == "PAGE-TOKEN"


def test_meta_keys_render_in_caller_order_and_change_the_hash() -> None:
    silent = build(
        _scripted().model(),
        documents=_notice({"page": "3", "section": "PAGE-TOKEN"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
    )
    forward = build(
        _scripted().model(),
        documents=_notice({"page": "3", "section": "PAGE-TOKEN"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        meta_keys=["page", "section"],
    )
    reverse = build(
        _scripted().model(),
        documents=_notice({"page": "3", "section": "PAGE-TOKEN"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        meta_keys=["section", "page"],
    )
    missing = build(
        _scripted().model(),
        documents=_notice({"page": "3", "section": "PAGE-TOKEN"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        meta_keys=["absent"],
    )
    empty = build(
        _scripted().model(),
        documents=_notice({"page": "3", "section": "PAGE-TOKEN"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        meta_keys=[],
    )
    assert '[meta {"page":"3","section":"PAGE-TOKEN"}]' in forward.rendered_documents
    assert '[meta {"section":"PAGE-TOKEN","page":"3"}]' in reverse.rendered_documents
    assert forward.prefix_hash != silent.prefix_hash
    assert reverse.prefix_hash != forward.prefix_hash
    assert missing.prefix_hash == silent.prefix_hash
    assert empty.prefix_hash == silent.prefix_hash
    assert "[meta " not in missing.rendered_documents


def test_meta_renderer_is_part_of_the_prefix() -> None:
    def render(document: Document, block: Block) -> str | None:
        page = block.meta.get("page")
        if not page or document.doc_id != "doc":
            return None
        return f"meta-page {page}"

    silent = build(
        _scripted().model(),
        documents=_notice({"page": "3\x00"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
    )
    shown = build(
        _scripted().model(),
        documents=_notice({"page": "3\x00"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        meta_renderer=render,
    )
    blank = build(
        _scripted().model(),
        documents=_notice({"page": "3"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        meta_renderer=lambda _document, _block: " \x00\n",
    )
    stripped = build(
        _scripted().model(),
        documents=_notice({"page": "3\x00"}),
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        meta_keys=["page"],
    )
    assert "meta-page 3" in shown.rendered_documents
    assert "\x00" not in shown.rendered_documents
    assert shown.prefix_hash != silent.prefix_hash
    assert blank.prefix_hash == silent.prefix_hash
    assert '[meta {"page":"3"}]' in stripped.rendered_documents
    assert "\x00" not in stripped.rendered_documents
    with pytest.raises(forkreduce.ForkReduceError, match="not both"):
        build(
            _scripted().model(),
            documents=_notice({"page": "3"}),
            instructions="rules",
            outputs=[Answer],
            warmup=False,
            meta_keys=["page"],
            meta_renderer=render,
        )
