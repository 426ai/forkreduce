"""Citation checks and field-limited repair."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from tests.support import Scripted, tool_result, user_text

from forkreduce import CitationCheck, CitationRepair, Query, build
from forkreduce.documents import Block, Document, coerce_documents
from forkreduce.policy import Policy


class Answer(BaseModel):
    """Grounded answer."""

    text: str
    quote: str
    block_id: str


class Item(BaseModel):
    note: str
    quote: str
    block_id: str


class Bundle(BaseModel):
    """Several citations."""

    items: list[Item]


DOC = "The library opens at nine."


def test_whitespace_normalized_quote_matches() -> None:
    documents = coerce_documents(Block(block_id="hours", text="The   library\nopens at nine."))
    check = CitationCheck()
    output = Answer(text="opens at nine", quote="The library opens at nine.", block_id="hours")
    assert check(output, documents=documents) == []


def test_repair_copies_only_allowed_fields_on_nested_items() -> None:
    original = Bundle(items=[Item(note="keep-me", quote="missing", block_id="nope")])
    updated = Bundle(items=[Item(note="changed", quote="The library opens at nine.", block_id="hours")])
    documents = coerce_documents(Block(block_id="hours", text=DOC))
    failures = CitationCheck()(original, documents=documents)
    assert len(failures) == 1
    repaired = CitationRepair().apply(original, updated, failures)
    assert repaired.items[0].note == "keep-me"
    assert repaired.items[0].quote == "The library opens at nine."
    assert repaired.items[0].block_id == "hours"
    assert original.items[0].quote == "missing"
    assert CitationCheck()(repaired, documents=documents) == []


@pytest.mark.asyncio
async def test_repair_continues_the_branch_and_ignores_other_fields() -> None:
    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        text = user_text(messages)
        if "could not be located" in text:
            return tool_result(
                info,
                "Answer",
                {
                    "text": "CHANGED",
                    "quote": DOC,
                    "block_id": "hours",
                },
            )
        return tool_result(
            info,
            "Answer",
            {"text": "original", "quote": "not in the document", "block_id": "hours"},
        )

    scripted = Scripted(handler)
    session = build(
        scripted.model(),
        documents=Document(
            doc_id="notice",
            title="Hours",
            blocks=[Block(block_id="hours", text=DOC)],
        ),
        instructions="cite the notice",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    result = await (
        session.map([Query(tail="When does it open?", output=Answer, id="q1")])
        .check(CitationCheck(), repair=CitationRepair(max_rounds=2))
        .run()
    )
    assert result.failures == []
    assert len(result.results) == 1
    output = result.results[0].output
    assert output is not None
    assert output.text == "original"
    assert output.quote == DOC
    assert result.results[0].needs_review is False
    assert "When does it open?" in user_text(scripted.calls[1][0])
    assert result.reports[0].action == "repaired"


@pytest.mark.asyncio
async def test_unfixed_citations_follow_the_failure_policy() -> None:
    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        return tool_result(
            info,
            "Answer",
            {"text": "original", "quote": "nowhere", "block_id": "hours"},
        )

    def _session():  # type: ignore[no-untyped-def]
        return build(
            Scripted(handler).model(),
            documents=Block(block_id="hours", text=DOC),
            instructions="rules",
            outputs=[Answer],
            warmup=False,
            policy=Policy(retries=0, retry_backoff=0, repair_failure="needs_review"),
        )

    reviewed = await (
        _session()
        .map([Query(tail="q", output=Answer, id="q1")])
        .check(CitationCheck(), repair=CitationRepair(max_rounds=1))
        .run()
    )
    assert reviewed.results[0].needs_review is True
    assert reviewed.results[0].output is not None
    assert reviewed.results[0].output.text == "original"
    assert reviewed.merged is None

    dropped = await (
        _session()
        .map([Query(tail="q", output=Answer, id="q1")])
        .check(CitationCheck(), repair=CitationRepair(max_rounds=0), on_failure="drop")
        .run()
    )
    assert dropped.results[0].dropped is True

    failed = await (
        _session()
        .map([Query(tail="q", output=Answer, id="q1")])
        .check(CitationCheck(), repair=CitationRepair(max_rounds=0), on_failure="fail")
        .run()
    )
    assert failed.failures
    assert failed.failures[0].code == "check_failed"
    assert failed.merged is None
