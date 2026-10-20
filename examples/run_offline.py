"""Offline walkthrough of forkreduce against a fictional library notice.

The model is a FunctionModel. It never calls a network and needs no API key.
Run from the repository root:

    python examples/run_offline.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic_ai import ModelResponse, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402
from pydantic_ai.usage import RequestUsage  # noqa: E402

from examples.notice import (  # noqa: E402
    INSTRUCTIONS,
    NOTICE,
    Answer,
    FindingList,
    LeadList,
)
from forkreduce import (  # noqa: E402
    Accept,
    CitationCheck,
    CitationRepair,
    DiscardExplanations,
    GapFill,
    Query,
    RequireReason,
    Scope,
    build,
)
from forkreduce.policy import Policy  # noqa: E402


def _user_text(messages: list[object]) -> str:
    chunks: list[str] = []
    for message in messages:
        for part in getattr(message, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                chunks.append(content)
    return "\n".join(chunks)


def _reply(info: object, name: str, payload: dict[str, object]) -> ModelResponse:
    tools = getattr(info, "output_tools", [])
    match = next(tool for tool in tools if tool.name == name)
    return ModelResponse(
        parts=[ToolCallPart(match.name, payload)],
        usage=RequestUsage(input_tokens=200, output_tokens=40, cache_read_tokens=180),
    )


def library_model() -> FunctionModel:
    """Stand-in model. Branch on the query tail, which is the only part that changes."""

    def respond(messages: list[object], info: object) -> ModelResponse:
        text = _user_text(messages)
        if "were not covered" in text:
            return _reply(
                info,
                "DiscardExplanations",
                {
                    "explanations": [
                        {
                            "item_id": "deadline",
                            "reason": "The deadline line is short enough for a clerk to confirm by hand.",
                        }
                    ]
                },
            )
        if "Scope delivery" in text:
            return _reply(
                info,
                "LeadList",
                {
                    "leads": [
                        {
                            "block_id": "titles",
                            "question": "Which titles are eligible?",
                        },
                        {
                            "block_id": "delivery",
                            "question": "Where should a sealed offer be delivered?",
                        },
                    ]
                },
            )
        if "Where should a sealed offer be delivered?" in text:
            return _reply(
                info,
                "Answer",
                {
                    "text": "Sealed envelopes go to the library front desk.",
                    "quote": ("Deliver sealed envelopes to the library front desk and mark them NB-LIB-2026-PER."),
                    "block_id": "delivery",
                },
            )
        if "INPUT_ITEMS" in text:
            ids = re.findall(r"(?m)^id: (\S+)", text)
            # The first merge deliberately skips the deadline input so the coverage
            # policy has something to explain. Later merges cite every id they see.
            if ids == ["titles", "quantities", "deadline"]:
                ids = ["titles", "quantities"]
            block_id = "delivery" if any("delivery" in item for item in ids) else "titles"
            return _reply(
                info,
                "FindingList",
                {
                    "items": [
                        {
                            "topic": "subscription notice",
                            "summary": "Facts cited from the frozen notice.",
                            "block_id": block_id,
                            "round": 1,
                            "source_ids": ids,
                        }
                    ]
                },
            )
        if "Which titles" in text:
            return _reply(
                info,
                "Answer",
                {
                    "text": "The reading room wants the Northbridge Science Digest and the River Almanac.",
                    "quote": (
                        "Eligible titles for the reading room are the Northbridge Science Digest and the River Almanac."
                    ),
                    "block_id": "titles",
                },
            )
        if "How many copies" in text:
            return _reply(
                info,
                "Answer",
                {
                    "text": "Forty copies of the digest and twenty-five of the almanac.",
                    "quote": (
                        "The library will take forty copies of the Northbridge Science Digest "
                        "and twenty-five copies of the River Almanac."
                    ),
                    "block_id": "quantities",
                },
            )
        if "When must offers" in text:
            return _reply(
                info,
                "Answer",
                {
                    "text": "Offers are due on 30 April 2026.",
                    "quote": "Offers must reach the front desk by 30 April 2026.",
                    "block_id": "deadline",
                },
            )
        raise AssertionError(f"offline model has no script for this tail:\n{text}")

    return FunctionModel(respond)


class ExplainUncovered:
    """Business policy: a missing input must be explained, not dropped quietly."""

    def decide(self, report: object) -> Accept | RequireReason:
        uncovered = tuple(getattr(report, "uncovered", ()))
        if uncovered:
            return RequireReason(uncovered)
        return Accept()


async def main() -> object:
    session = build(
        library_model(),
        documents=NOTICE,
        instructions=INSTRUCTIONS,
        outputs=[Answer, LeadList, FindingList, DiscardExplanations],
        warmup=True,
        policy=Policy(concurrency=4, retries=1, retry_backoff=0),
    )
    gap = GapFill(
        lead_output=LeadList,
        item_output=Answer,
        scopes=[Scope("delivery", "how a sealed offer is delivered")],
        instructions="Report blocks in this scope that the merge does not already cite.",
    )
    result = await (
        session.map(
            [
                Query(tail="Which titles are eligible?", output=Answer, id="titles"),
                Query(tail="How many copies are requested?", output=Answer, id="quantities"),
                Query(tail="When must offers arrive?", output=Answer, id="deadline"),
            ]
        )
        .check(CitationCheck(), repair=CitationRepair(max_rounds=2))
        .filter(lambda item: bool(item.quote))
        .reduce(output=FindingList, instructions="Merge the notice into one list.", coverage=ExplainUncovered())
        .followup(gap, max_rounds=2)
        .run()
    )
    print(f"prefix {session.prefix_hash[:12]}")
    print(f"rounds {len(result.rounds)}  cache_hit_rate {result.cache_hit_rate:.2f}")
    print(f"failures {len(result.failures)}")
    for report in result.reports:
        uncovered = getattr(report, "uncovered", None)
        reasons = getattr(report, "reasons", None)
        if uncovered is not None:
            print(f"coverage uncovered={list(uncovered)} reasons={reasons}")
    if result.merged is not None:
        for item in result.merged.items:
            print(f"round {item.round} block {item.block_id} sources {item.source_ids}: {item.summary}")
    return result


if __name__ == "__main__":
    asyncio.run(main())
