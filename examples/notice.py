"""Fictional periodical notice used by the offline and live examples.

Northbridge is not a real city, and none of these titles or quantities come from
a real procurement.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from forkreduce import CITATION_INSTRUCTIONS, Block, Citation, Document, Traced

NOTICE = Document(
    doc_id="nb-lib-2026-per",
    title="Northbridge Public Library 2026 periodical subscription notice",
    blocks=[
        Block(
            block_id="purpose",
            text=("Northbridge Public Library invites sealed offers for its 2026 periodical subscriptions."),
        ),
        Block(
            block_id="titles",
            text=("Eligible titles for the reading room are the Northbridge Science Digest and the River Almanac."),
        ),
        Block(
            block_id="quantities",
            text=(
                "The library will take forty copies of the Northbridge Science Digest "
                "and twenty-five copies of the River Almanac."
            ),
        ),
        Block(
            block_id="deadline",
            text="Offers must reach the front desk by 30 April 2026.",
        ),
        Block(
            block_id="delivery",
            text=("Deliver sealed envelopes to the library front desk and mark them NB-LIB-2026-PER."),
        ),
    ],
)

INSTRUCTIONS = (
    "Answer only from the frozen notice. Cite a quote and its block_id. "
    "When merging, every row needs source_ids from the input list. "
    "Do not invent titles, dates, or quantities. " + CITATION_INSTRUCTIONS
)


class Answer(Citation):
    """One fact taken from the notice."""

    text: str


class Lead(BaseModel):
    """A place in the notice the merge might have skipped."""

    block_id: str
    question: str


class LeadList(BaseModel):
    """Gap-fill leads."""

    leads: list[Lead]


class Finding(Traced):
    """A merged fact. ``round`` is stamped by the plan."""

    topic: str
    summary: str
    block_id: str = ""
    round: int = 0
    source_ids: list[str] = Field(default_factory=list)


class FindingList(BaseModel):
    """The merged reading of the notice."""

    items: list[Finding]
