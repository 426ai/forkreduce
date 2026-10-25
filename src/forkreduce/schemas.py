"""Output models the framework understands.

``Traced`` is the provenance contract: a reduce output that cites input ids.
``DiscardExplanations`` is the payload for :class:`forkreduce.policy.RequireReason`.
Register it in ``outputs`` at :func:`forkreduce.build` time if a coverage policy
will ask for reasons. It is not added later, because that would change the tool set.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

CITATION_INSTRUCTIONS = (
    "Each quote must be a contiguous passage contained entirely in the single block named by block_id. "
    "Text appearing elsewhere in the document is not evidence for that block. "
    "If evidence crosses block boundaries, split it into separate citations, each with its own block_id."
)


class Citation(BaseModel):
    """One passage from one referenced block; validate it with CitationCheck."""

    quote: str = Field(
        description=(
            "A contiguous quote contained entirely in the block named by block_id. "
            "Do not join passages across blocks; split them into separate citations."
        )
    )
    block_id: str = Field(
        description=(
            "The ID of the single block containing the entire quote, not a document ID "
            "or a different block where only part of the quote appears."
        )
    )


class Traced(BaseModel):
    """Mixin for a merged entry that must cite the inputs it came from."""

    source_ids: list[str] = Field(default_factory=list)


class DiscardExplanation(BaseModel):
    """Why one input item was left out of a merge."""

    item_id: str
    reason: str


class DiscardExplanations(BaseModel):
    """Model output for a RequireReason continuation. Register this on the session."""

    explanations: list[DiscardExplanation]
