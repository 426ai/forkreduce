"""Output models the framework understands.

``Traced`` is the provenance contract: a reduce output that cites input ids.
``DiscardExplanations`` is the payload for :class:`forkreduce.policy.RequireReason`.
Register it in ``outputs`` at :func:`forkreduce.build` time if a coverage policy
will ask for reasons. It is not added later, because that would change the tool set.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
