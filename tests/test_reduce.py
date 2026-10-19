"""Semantic reduce, coverage actions, follow-up rounds, and gap fill."""

from __future__ import annotations

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
    reason_text = user_text(scripted.calls[-1][0])
    assert "duplicate" not in reason_text or "were not covered" in reason_text
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
    reduce_prompts = [user_text(messages) for messages, _info in scripted.calls if "INPUT_ITEMS" in user_text(messages)]
    assert reduce_prompts
    assert "id: keep" in reduce_prompts[-1]
    assert "id: drop" not in reduce_prompts[-1]
    assert filtered.merged is not None
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
