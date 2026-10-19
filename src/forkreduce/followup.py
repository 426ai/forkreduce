"""The hook between rounds, and the built-in gap-fill strategy.

``followup`` receives the round's merged output and query results. It may call
the model (another query on the same prefix) or stay a pure function. An empty
list ends the run, as does reaching ``max_rounds``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from forkreduce.coverage import ReduceReport
from forkreduce.failures import Failure, FailureKind
from forkreduce.policy import Accept, Action, Drop, Fail
from forkreduce.query import Query, QueryResult

FollowupFn = Callable[["RoundState", Any], Awaitable[list[Query[Any]]] | list[Query[Any]]]


@dataclass
class RoundState:
    """What a followup hook can see after one map/check/reduce round.

    ``side_results`` collects model calls the hook makes (gap-fill queries, for
    example) so their usage is not dropped. ``failures`` is the same idea for
    errors the hook observed.
    """

    round: int
    results: list[QueryResult[Any]]
    outputs: list[Any]
    merged: Any
    report: ReduceReport | None
    failures: list[Failure] = field(default_factory=list)
    side_results: list[QueryResult[Any]] = field(default_factory=list)

    def add_result(self, result: QueryResult[Any]) -> None:
        self.side_results.append(result)
        if result.failure is not None:
            self.failures.append(result.failure)


@dataclass(frozen=True)
class Scope:
    """One slice of the document a gap-fill query is responsible for."""

    name: str
    responsibility: str


@dataclass
class LeadReport:
    """Positions a gap-fill pass already covered, and the new leads it found."""

    kind: str
    new_positions: list[str]
    duplicate_positions: list[str]
    leads: list[Any]


class LeadPolicy(Protocol):
    """Decide which new leads become the next round's queries."""

    def decide(self, report: LeadReport) -> Action:
        """Return an action for ``report``."""
        ...


class AcceptLeads:
    """Default lead policy: keep every lead whose position is not already covered."""

    def decide(self, report: LeadReport) -> Action:
        return Accept()


@dataclass
class GapFill:
    """One gap-fill query per scope, then a query per new lead.

    Leads are deduped by position (``block_id`` by default) against the merged
    output and against each other. The gap-fill query itself runs inside the
    hook, on the same session, so it shares the prefix cache. Its
    :class:`QueryResult` is attached to the round state.

    ``lead_output`` and ``item_output`` must both be registered on the session.
    """

    lead_output: type[BaseModel]
    item_output: type[BaseModel]
    scopes: Sequence[Scope] | None = None
    instructions: str = (
        "Look for requirements in your scope that the merged list does not already cover. "
        "Return leads only for those gaps."
    )
    policy: LeadPolicy | None = None
    lead_items: Callable[[Any], Sequence[Any]] | None = None
    lead_position: Callable[[Any], str] | None = None
    covered_positions: Callable[[Any], Sequence[str]] | None = None
    query_for_lead: Callable[[Any, RoundState], Query[Any]] | None = None

    async def __call__(self, state: RoundState, session: Any) -> list[Query[Any]]:
        scopes = list(self.scopes) if self.scopes else [Scope("all", "the whole document")]
        covered = set(self._covered(state.merged))
        fresh: list[Any] = []
        duplicates: list[str] = []
        for scope in scopes:
            query = Query(
                tail=self._gap_tail(state.merged, scope),
                output=self.lead_output,
                id=f"gapfill-r{state.round}-{scope.name}",
                round=state.round,
            )
            result = await session.ask(query)
            state.add_result(result)
            if result.failure is not None or result.output is None:
                continue
            for lead in self._leads(result.output):
                position = self._position(lead)
                if not position or position in covered:
                    duplicates.append(position or "")
                    continue
                covered.add(position)
                fresh.append(lead)
        report = LeadReport(
            kind="leads",
            new_positions=[self._position(lead) for lead in fresh],
            duplicate_positions=[item for item in duplicates if item],
            leads=list(fresh),
        )
        action = (self.policy or AcceptLeads()).decide(report)
        if isinstance(action, Fail):
            state.failures.append(
                Failure(
                    kind=FailureKind.PERMANENT,
                    message=action.reason,
                    retryable=False,
                    code="leads_rejected",
                )
            )
            return []
        if isinstance(action, Drop):
            banned = set(action.items)
            fresh = [lead for lead in fresh if self._position(lead) not in banned]
        elif not isinstance(action, Accept):
            state.failures.append(
                Failure(
                    kind=FailureKind.PERMANENT,
                    message=f"lead policy returned {type(action).__name__}, which gap-fill cannot run",
                    retryable=False,
                    code="bad_action",
                )
            )
            return []
        return [self._to_query(lead, state) for lead in fresh]

    def _leads(self, output: Any) -> Sequence[Any]:
        if self.lead_items is not None:
            return list(self.lead_items(output))
        for name in ("leads", "items"):
            value = getattr(output, name, None)
            if isinstance(value, list):
                return value
        return []

    def _position(self, lead: Any) -> str:
        if self.lead_position is not None:
            return self.lead_position(lead)
        block_id = getattr(lead, "block_id", None)
        if isinstance(block_id, str):
            return block_id
        return ""

    def _covered(self, merged: Any) -> list[str]:
        if self.covered_positions is not None:
            return list(self.covered_positions(merged))
        return sorted(_collect_block_ids(merged))

    def _to_query(self, lead: Any, state: RoundState) -> Query[Any]:
        if self.query_for_lead is not None:
            return self.query_for_lead(lead, state)
        question = getattr(lead, "question", None) or getattr(lead, "text", None) or str(lead)
        position = self._position(lead) or "lead"
        return Query(
            id=f"gap-r{state.round}-{position}",
            tail=str(question),
            output=self.item_output,
            meta={"block_id": position, "gap": True},
            round=state.round + 1,
        )

    def _gap_tail(self, merged: Any, scope: Scope) -> str:
        if isinstance(merged, BaseModel):
            rendered = merged.model_dump_json()
        else:
            rendered = "null" if merged is None else str(merged)
        return f"Scope {scope.name}: {scope.responsibility}\n{self.instructions.strip()}\n\nAlready merged:\n{rendered}"


def _collect_block_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, BaseModel):
        if "block_id" in type(value).model_fields:
            block_id = getattr(value, "block_id")  # noqa: B009 - field is optional on BaseModel
            if isinstance(block_id, str) and block_id:
                found.add(block_id)
        for name in type(value).model_fields:
            found |= _collect_block_ids(getattr(value, name))
    elif isinstance(value, list):
        for item in value:
            found |= _collect_block_ids(item)
    return found
