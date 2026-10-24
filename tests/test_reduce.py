"""Semantic reduce, coverage actions, follow-up rounds, and gap fill."""

from __future__ import annotations

import json
import re

import pytest
from pydantic import BaseModel
from tests.support import Scripted, tool_result, user_text

from forkreduce import (
    Accept,
    DiscardExplanations,
    Fail,
    GapFill,
    Provenance,
    Query,
    RequireReason,
    Retry,
    Scope,
    Traced,
    build,
)
from forkreduce.policy import Drop, Policy
from forkreduce.reduce import ReduceItem


class Answer(BaseModel):
    """Grounded answer."""

    text: str
    quote: str
    block_id: str


class Finding(Traced):
    """One merged row."""

    topic: str
    summary: str
    block_id: str
    round: int = 0


class FindingList(BaseModel):
    """Merged findings."""

    items: list[Finding]


class Lead(BaseModel):
    block_id: str
    question: str


class LeadList(BaseModel):
    """Possible gaps."""

    leads: list[Lead]


def _ids(text: str) -> list[str]:
    return re.findall(r"(?m)^id: (\S+)", text)


def _finding_list(ids: list[str], block_id: str = "b1") -> dict[str, object]:
    return {
        "items": [
            {
                "topic": "row",
                "summary": "merged",
                "block_id": block_id,
                "round": 0,
                "source_ids": ids,
            }
        ]
    }


@pytest.mark.asyncio
async def test_provenance_coverage_actions_and_tree_reduce() -> None:
    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        text = user_text(messages)
        if "were not covered" in text:
            return tool_result(
                info,
                "DiscardExplanations",
                {"explanations": [{"item_id": "b", "reason": "duplicate of a"}]},
            )
        ids = _ids(text)
        if "include everyone" in text:
            return tool_result(info, "FindingList", _finding_list(ids))
        if "omit-b" in text:
            return tool_result(info, "FindingList", _finding_list([item for item in ids if item != "b"]))
        if "force-invent" in text:
            return tool_result(info, "FindingList", _finding_list([*ids, "nope"]))
        return tool_result(info, "FindingList", _finding_list(ids))

    scripted = Scripted(handler)
    session = build(
        scripted.model(),
        documents="alpha beta",
        instructions="rules",
        outputs=[FindingList, DiscardExplanations, Answer],
        warmup=False,
        policy=Policy(retries=1, retry_backoff=0),
    )
    items = [
        Answer(text="one", quote="alpha", block_id="b1"),
        Answer(text="two", quote="beta", block_id="b1"),
    ]

    accepted = await session.reduce(
        items,
        output=FindingList,
        instructions="omit-b",
        id_of=lambda item: "a" if item.text == "one" else "b",
        coverage=type("P", (), {"decide": staticmethod(lambda report: Accept())})(),
    )
    assert accepted.report.uncovered == ["b"]
    assert accepted.report.covered == ["a"]
    assert accepted.fatal is False
    assert accepted.output is not None

    class Explain:
        def decide(self, report):  # type: ignore[no-untyped-def]
            if report.uncovered:
                return RequireReason(tuple(report.uncovered))
            return Accept()

    explained = await session.reduce(
        items,
        output=FindingList,
        instructions="omit-b",
        id_of=lambda item: "a" if item.text == "one" else "b",
        coverage=Explain(),
    )
    assert explained.action == "RequireReason"
    assert explained.reasons and explained.reasons[0].item_id == "b"
    assert explained.reasons[0].reason == "duplicate of a"
    assert len(scripted.calls[-1][0]) > 1

    class Once:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, report):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return Retry((), "include everyone")
            return Accept()

    retried = await session.reduce(
        items,
        output=FindingList,
        instructions="omit-b",
        id_of=lambda item: "a" if item.text == "one" else "b",
        coverage=Once(),
        max_rounds=3,
    )
    assert retried.report.uncovered == []
    assert retried.failure is None
    assert any("include everyone" in user_text(messages) for messages, _info in scripted.calls)

    failed = await session.reduce(
        items,
        output=FindingList,
        id_of=lambda item: "a" if item.text == "one" else "b",
        coverage=type("F", (), {"decide": staticmethod(lambda _report: Fail("no gaps allowed"))})(),
    )
    assert failed.fatal is True
    assert failed.failure is not None and failed.failure.code == "coverage_failed"

    # Unknown source ids are retried, then still reported if the model keeps inventing them.
    invented = await session.reduce(
        items,
        output=FindingList,
        instructions="force-invent",
        id_of=lambda item: "a" if item.text == "one" else "b",
        coverage=type("P", (), {"decide": staticmethod(lambda report: Accept())})(),
    )
    assert "nope" in invented.report.invalid_refs

    tree = await session.reduce(
        [Answer(text=str(i), quote="alpha", block_id="b1") for i in range(4)],
        output=FindingList,
        max_batch=2,
        id_of=lambda item: f"q{item.text}",
    )
    assert tree.failure is None
    assert tree.output is not None
    cited = set(tree.output.items[0].source_ids)
    assert cited == {"q0", "q1", "q2", "q3"}
    assert len(tree.calls) == 3


@pytest.mark.asyncio
async def test_followup_rounds_filter_and_gap_fill() -> None:
    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        text = user_text(messages)
        if "Scope" in text:
            leads = [{"block_id": "b1", "question": "repeat"}]
            if "delivery" in text:
                leads.append({"block_id": "b5", "question": "Where is the desk?"})
            if "hours" in text:
                leads.append({"block_id": "b6", "question": "When does it open?"})
            return tool_result(info, "LeadList", {"leads": leads})
        if "Where is the desk?" in text:
            return tool_result(
                info,
                "Answer",
                {"text": "front desk", "quote": "front desk", "block_id": "b5"},
            )
        if "INPUT_ITEMS" in text:
            return tool_result(info, "FindingList", _finding_list(_ids(text), block_id="b1"))
        label = "drop" if "\ndrop\n" in f"\n{text}\n" else "keep"
        return tool_result(
            info,
            "Answer",
            {"text": label, "quote": "alpha", "block_id": "b1"},
        )

    def session_for(scripted: Scripted):  # type: ignore[no-untyped-def]
        return build(
            scripted.model(),
            documents="alpha beta",
            instructions="rules",
            outputs=[Answer, FindingList, LeadList, DiscardExplanations],
            warmup=False,
            policy=Policy(retries=0, retry_backoff=0),
        )

    session = session_for(Scripted(handler))

    async def more(state, _session):  # type: ignore[no-untyped-def]
        if state.round >= 2:
            return []
        return [Query(tail=f"extra-{state.round}", output=Answer, id=f"extra-{state.round}")]

    # max_rounds counts map rounds. followup is not called once the cap is reached.
    capped = await session.map([Query(tail="seed", output=Answer, id="seed")]).followup(more, max_rounds=2).run()
    assert [record.round for record in capped.rounds] == [1, 2]
    assert any(result.query_id == "extra-1" for result in capped.results)
    assert all(result.query_id != "extra-2" for result in capped.results)

    async def stop(_state, _session):  # type: ignore[no-untyped-def]
        return []

    stopped = await session.map([Query(tail="seed", output=Answer, id="seed")]).followup(stop, max_rounds=4).run()
    assert len(stopped.rounds) == 1

    scripted = Scripted(handler)
    session = session_for(scripted)
    filtered = await (
        session.map(
            [
                Query(tail="keep", output=Answer, id="keep"),
                Query(tail="drop", output=Answer, id="drop"),
            ]
        )
        .filter(lambda item: item.text != "drop")
        .reduce(output=FindingList)
        .run()
    )
    assert filtered.merged is not None
    assert filtered.merged.items[0].source_ids == ["keep"]
    assert filtered.merged.items[0].round == 1

    class DropNew:
        def decide(self, report):  # type: ignore[no-untyped-def]
            return Drop(tuple(report.new_positions))

    scripted = Scripted(handler)
    session = session_for(scripted)
    gap = GapFill(
        lead_output=LeadList,
        item_output=Answer,
        scopes=[Scope("delivery", "where to deliver"), Scope("hours", "when it opens")],
    )
    filled = await (
        session.map([Query(tail="keep", output=Answer, id="keep")])
        .reduce(output=FindingList)
        .followup(gap, max_rounds=2)
        .run()
    )
    assert len(filled.rounds) == 2
    follow_ids = {result.query_id for result in filled.rounds[1].results}
    assert "gap-r1-b5" in follow_ids
    assert "gap-r1-b6" in follow_ids
    assert "gap-r1-b1" not in follow_ids
    assert filled.usage.cache_hit_tokens > 0
    assert Provenance().evaluate(filled.merged, [item.id for item in []]).uncovered == []

    scripted = Scripted(handler)
    session = session_for(scripted)
    blocked = GapFill(
        lead_output=LeadList,
        item_output=Answer,
        scopes=[Scope("delivery", "where to deliver")],
        policy=DropNew(),
    )
    held = await (
        session.map([Query(tail="keep", output=Answer, id="keep")])
        .reduce(output=FindingList)
        .followup(blocked, max_rounds=3)
        .run()
    )
    assert len(held.rounds) == 1


def _per_input(ids: list[str]) -> dict[str, object]:
    return {
        "items": [
            {
                "topic": "row",
                "summary": item,
                "block_id": "b1",
                "round": 0,
                "source_ids": [item],
            }
            for item in ids
        ]
    }


@pytest.mark.asyncio
async def test_tree_reduce_shrinks_or_stops() -> None:
    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        return tool_result(info, "FindingList", _per_input(_ids(user_text(messages))))

    session = build(
        Scripted(handler).model(),
        documents="alpha",
        instructions="rules",
        outputs=[FindingList, Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    stuck = await session.reduce(
        [Answer(text="0", quote="alpha", block_id="b1"), Answer(text="1", quote="alpha", block_id="b1")],
        output=FindingList,
        max_batch=1,
        id_of=lambda item: f"q{item.text}",
    )
    assert stuck.failure is not None
    assert stuck.failure.code == "reduce_did_not_shrink"
    assert len(stuck.calls) == 2

    finished = await session.reduce(
        [Answer(text=str(i), quote="alpha", block_id="b1") for i in range(4)],
        output=FindingList,
        max_batch=2,
        id_of=lambda item: f"q{item.text}",
    )
    assert finished.failure is None
    assert finished.output is not None
    cited: set[str] = set()
    for item in finished.output.items:
        cited.update(item.source_ids)
    assert cited == {"q0", "q1", "q2", "q3"}
    assert len(finished.calls) == 3


@pytest.mark.asyncio
async def test_selective_retry_keeps_previously_covered_rows() -> None:
    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        text = user_text(messages)
        ids = _ids(text)
        if "only-b" in text:
            return tool_result(info, "FindingList", _finding_list([item for item in ids if item == "b"]))
        if "omit-b" in text:
            return tool_result(info, "FindingList", _finding_list([item for item in ids if item != "b"]))
        return tool_result(info, "FindingList", _finding_list(ids))

    session = build(
        Scripted(handler).model(),
        documents="alpha beta",
        instructions="rules",
        outputs=[FindingList, Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )

    class Once:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, report):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return Retry(("b",), "only-b")
            return Accept()

    outcome = await session.reduce(
        [Answer(text="one", quote="alpha", block_id="b1"), Answer(text="two", quote="beta", block_id="b1")],
        output=FindingList,
        instructions="omit-b",
        id_of=lambda item: "a" if item.text == "one" else "b",
        coverage=Once(),
        max_rounds=3,
    )
    assert outcome.failure is None
    assert outcome.report.covered == ["a", "b"]
    assert outcome.report.uncovered == []
    assert outcome.output is not None
    cited = {source for item in outcome.output.items for source in item.source_ids}
    assert cited == {"a", "b"}


@pytest.mark.asyncio
async def test_coverage_retries_accumulate_calls() -> None:
    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        text = user_text(messages)
        ids = _ids(text)
        if "include everyone" in text:
            return tool_result(info, "FindingList", _finding_list(ids))
        return tool_result(info, "FindingList", _finding_list([item for item in ids if item != "b"]))

    session = build(
        Scripted(handler).model(),
        documents="alpha beta",
        instructions="rules",
        outputs=[FindingList, Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )

    class Once:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, _report):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return Retry((), "include everyone")
            return Accept()

    outcome = await session.reduce(
        [Answer(text="one", quote="alpha", block_id="b1"), Answer(text="two", quote="beta", block_id="b1")],
        output=FindingList,
        instructions="omit-b",
        id_of=lambda item: "a" if item.text == "one" else "b",
        coverage=Once(),
        max_rounds=3,
    )
    assert outcome.failure is None
    assert len(outcome.calls) == 2
    assert outcome.result is not None
    assert outcome.result.usage.input_tokens == 200


@pytest.mark.asyncio
async def test_require_reason_must_name_every_requested_id() -> None:
    mode = {"bad": "empty"}

    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        text = user_text(messages)
        if "were not covered" in text:
            if "previous explanation was rejected" in text or mode["bad"] == "good":
                return tool_result(
                    info,
                    "DiscardExplanations",
                    {"explanations": [{"item_id": "b", "reason": "duplicate of a"}]},
                )
            if mode["bad"] == "empty":
                return tool_result(info, "DiscardExplanations", {"explanations": []})
            if mode["bad"] == "unrelated":
                return tool_result(
                    info,
                    "DiscardExplanations",
                    {"explanations": [{"item_id": "nope", "reason": "no"}]},
                )
            if mode["bad"] == "duplicate":
                return tool_result(
                    info,
                    "DiscardExplanations",
                    {
                        "explanations": [
                            {"item_id": "b", "reason": "one"},
                            {"item_id": "b", "reason": "two"},
                        ]
                    },
                )
            return tool_result(
                info,
                "DiscardExplanations",
                {"explanations": [{"item_id": "b", "reason": "   "}]},
            )
        ids = _ids(text)
        return tool_result(info, "FindingList", _finding_list([item for item in ids if item != "b"]))

    def session_for(retries: int):  # type: ignore[no-untyped-def]
        return build(
            Scripted(handler).model(),
            documents="alpha beta",
            instructions="rules",
            outputs=[FindingList, DiscardExplanations, Answer],
            warmup=False,
            policy=Policy(retries=retries, retry_backoff=0),
        )

    class Explain:
        def decide(self, report):  # type: ignore[no-untyped-def]
            if report.uncovered:
                return RequireReason(tuple(report.uncovered))
            return Accept()

    mode["bad"] = "empty"
    recovered = await session_for(1).reduce(
        [Answer(text="one", quote="alpha", block_id="b1"), Answer(text="two", quote="beta", block_id="b1")],
        output=FindingList,
        instructions="omit-b",
        id_of=lambda item: "a" if item.text == "one" else "b",
        coverage=Explain(),
    )
    assert recovered.failure is None
    assert recovered.reasons and recovered.reasons[0].item_id == "b"
    reason_calls = [
        call
        for call in recovered.calls
        if call.output is not None and type(call.output).__name__ == "DiscardExplanations"
    ]
    assert len(reason_calls) == 2

    for bad in ("unrelated", "duplicate", "blank"):
        mode["bad"] = bad
        failed = await session_for(0).reduce(
            [Answer(text="one", quote="alpha", block_id="b1"), Answer(text="two", quote="beta", block_id="b1")],
            output=FindingList,
            instructions="omit-b",
            id_of=lambda item: "a" if item.text == "one" else "b",
            coverage=Explain(),
        )
        assert failed.failure is not None
        assert failed.failure.code == "reasons_incomplete"
        assert failed.fatal is True


@pytest.mark.asyncio
async def test_drop_is_not_a_reduce_action() -> None:
    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        return tool_result(info, "FindingList", _finding_list(_ids(user_text(messages))))

    session = build(
        Scripted(handler).model(),
        documents="alpha",
        instructions="rules",
        outputs=[FindingList, Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    outcome = await session.reduce(
        [Answer(text="one", quote="alpha", block_id="b1")],
        output=FindingList,
        id_of=lambda _item: "a",
        coverage=type("D", (), {"decide": staticmethod(lambda _report: Drop(("a",)))})(),
    )
    assert outcome.fatal is True
    assert outcome.failure is not None
    assert outcome.failure.code == "bad_action"


@pytest.mark.asyncio
async def test_round_record_includes_reduce_failure() -> None:
    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        text = user_text(messages)
        if "INPUT_ITEMS" in text:
            return tool_result(info, "FindingList", _finding_list(_ids(text)))
        return tool_result(info, "Answer", {"text": "keep", "quote": "alpha", "block_id": "b1"})

    session = build(
        Scripted(handler).model(),
        documents="alpha",
        instructions="rules",
        outputs=[Answer, FindingList],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    result = await (
        session.map([Query(tail="keep", output=Answer, id="keep")])
        .reduce(
            output=FindingList,
            coverage=type("F", (), {"decide": staticmethod(lambda _report: Fail("no gaps"))})(),
        )
        .run()
    )
    assert any(failure.code == "coverage_failed" for failure in result.failures)
    assert any(failure.code == "coverage_failed" for failure in result.rounds[0].failures)


class QuestionFinding(Traced):
    question_id: str
    answer: str


class QuestionFindings(BaseModel):
    answers: list[QuestionFinding]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_batch", [2, 200])
async def test_nested_question_sources_use_current_fork_ids_and_expand_tree_aliases(max_batch: int) -> None:
    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        text = user_text(messages)
        match = re.search(r"(?m)^ALLOWED_SOURCE_IDS_BY_QUESTION: (.+)$", text)
        mapping = json.loads(match.group(1)) if match else {}
        return tool_result(
            info,
            "QuestionFindings",
            {
                "answers": [
                    {"question_id": question, "answer": "merged", "source_ids": sources}
                    for question, sources in mapping.items()
                ]
            },
        )

    session = build(
        Scripted(handler).model(),
        documents="alpha beta",
        instructions="rules",
        outputs=[QuestionFindings],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    items = [
        ReduceItem(
            id="fork:0",
            value=QuestionFindings(
                answers=[QuestionFinding(question_id="opening", answer="alpha", source_ids=["document"])]
            ),
        ),
        ReduceItem(
            id="fork:1",
            value={"nested": [{"question_id": "opening", "source_ids": ["b1"]}, {"question_id": "opening"}]},
        ),
        ReduceItem(id="fork:2", value=({"question_id": "closing"},)),
        ReduceItem(id="fork:3", value={"nested": {"question_id": "closing"}}),
    ]
    frozen_hash = session.prefix_hash
    outcome = await session.reduce(items, output=QuestionFindings, max_batch=max_batch)
    assert session.prefix_hash == frozen_hash
    assert all(call.prefix_hash == frozen_hash for call in outcome.calls)
    assert outcome.failure is None
    assert outcome.report.invalid_refs == []
    assert outcome.report.uncovered == []
    assert outcome.output is not None
    by_question = {row.question_id: set(row.source_ids) for row in outcome.output.answers}
    assert by_question == {
        "opening": {"fork:0", "fork:1"},
        "closing": {"fork:2", "fork:3"},
    }
    assert len(outcome.calls) == (3 if max_batch == 2 else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_source", ["document", "b1"])
async def test_document_and_block_ids_are_still_invalid_reduce_sources(invalid_source: str) -> None:
    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        return tool_result(info, "FindingList", _finding_list(["fork:0", invalid_source]))

    session = build(
        Scripted(handler).model(),
        documents="alpha",
        instructions="rules",
        outputs=[FindingList],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    outcome = await session.reduce([ReduceItem(id="fork:0", value={"text": "alpha"})], output=FindingList)
    assert outcome.report.covered == ["fork:0"]
    assert outcome.report.invalid_refs == [invalid_source]
