"""Coverage reports and provenance.

The framework computes which input ids were cited. A caller policy decides what
to do about the gaps. The default policy accepts the merge and leaves every
uncovered id on the report, so nothing disappears quietly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from forkreduce.policy import Accept, Action


@dataclass
class ReduceReport:
    """Coverage of one reduce against a set of input ids.

    ``covered`` and ``uncovered`` partition the input ids. ``invalid_refs`` are
    source ids that were not in the input. ``untraced`` are output paths whose
    ``source_ids`` were missing or empty. ``reasons`` is filled when a policy
    returns :class:`forkreduce.policy.RequireReason`.
    """

    covered: list[str]
    uncovered: list[str]
    invalid_refs: list[str]
    untraced: list[str] = field(default_factory=list)
    reasons: list[Any] = field(default_factory=list)
    kind: str = "reduce"

    @property
    def ok(self) -> bool:
        return not self.uncovered and not self.invalid_refs and not self.untraced


@dataclass
class TracedNode:
    """One object in a reduce output that carries ``source_ids``."""

    path: str
    source_ids: list[str]
    value: Any


class ProvenanceCheck(Protocol):
    """Check that an output cites real input ids and does not invent entries."""

    def evaluate(self, output: Any, input_ids: Sequence[str]) -> ReduceReport:
        """Return coverage of ``output`` against ``input_ids``."""
        ...


class Provenance:
    """Walk ``source_ids`` and compare them to the ids that were actually given.

    Any pydantic model with a ``source_ids: list`` field counts, including
    subclasses of :class:`forkreduce.schemas.Traced`. An output with no such
    field is reported as untraced rather than treated as a full cover.
    """

    def evaluate(self, output: Any, input_ids: Sequence[str]) -> ReduceReport:
        allowed = set(input_ids)
        nodes = iter_traced(output)
        if not nodes:
            return ReduceReport(
                covered=[],
                uncovered=list(input_ids),
                invalid_refs=[],
                untraced=["$"],
            )
        cited: list[str] = []
        invalid: list[str] = []
        untraced: list[str] = []
        for node in nodes:
            if not node.source_ids:
                untraced.append(node.path)
                continue
            for source_id in node.source_ids:
                if source_id in allowed:
                    cited.append(source_id)
                else:
                    invalid.append(source_id)
        cited_set = set(cited)
        covered = [item_id for item_id in input_ids if item_id in cited_set]
        uncovered = [item_id for item_id in input_ids if item_id not in cited_set]
        return ReduceReport(
            covered=covered,
            uncovered=uncovered,
            invalid_refs=_unique(invalid),
            untraced=untraced,
        )


class AcceptUncovered:
    """Default coverage policy: accept, and keep uncovered ids on the report."""

    def decide(self, report: ReduceReport) -> Action:
        return Accept()


class CoveragePolicy(Protocol):
    """Business decision for a coverage report. The framework executes the action."""

    def decide(self, report: ReduceReport) -> Action:
        """Return the action to take for ``report``."""
        ...


def iter_traced(value: Any, prefix: str = "") -> list[TracedNode]:
    """Find models that declare ``source_ids``."""
    found: list[TracedNode] = []
    if isinstance(value, BaseModel) and _has_source_ids(value):
        raw = getattr(value, "source_ids")  # noqa: B009 - field is optional on BaseModel
        source_ids = [str(item) for item in raw] if isinstance(raw, list) else []
        found.append(TracedNode(prefix or "$", source_ids, value))
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            if name == "source_ids":
                continue
            child = getattr(value, name)
            child_prefix = name if not prefix else f"{prefix}.{name}"
            found.extend(iter_traced(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(iter_traced(child, f"{prefix}[{index}]"))
    return found


def expand_id(source_id: str, alias: dict[str, list[str]], seen: set[str] | None = None) -> list[str]:
    """Expand a batch id back to the original ids it stands for."""
    if source_id not in alias:
        return [source_id]
    seen = set() if seen is None else set(seen)
    if source_id in seen:
        return []
    seen.add(source_id)
    expanded: list[str] = []
    for child in alias[source_id]:
        expanded.extend(expand_id(child, alias, seen))
    return expanded


def rewrite_source_ids(value: Any, alias: dict[str, list[str]]) -> Any:
    """Replace batch ids in every ``source_ids`` list with the original ids."""
    if not alias:
        return value
    if isinstance(value, BaseModel):
        updates: dict[str, Any] = {}
        for name in type(value).model_fields:
            current = getattr(value, name)
            if name == "source_ids" and isinstance(current, list):
                expanded: list[str] = []
                for source_id in current:
                    expanded.extend(expand_id(str(source_id), alias))
                deduped = _unique(expanded)
                if deduped != [str(item) for item in current]:
                    updates[name] = deduped
                continue
            rewritten = rewrite_source_ids(current, alias)
            if rewritten is not current:
                updates[name] = rewritten
        if updates:
            return value.model_copy(update=updates)
        return value
    if isinstance(value, list):
        rewritten_list = [rewrite_source_ids(item, alias) for item in value]
        if any(new is not old for new, old in zip(rewritten_list, value, strict=False)):
            return rewritten_list
        return value
    return value


def _has_source_ids(value: BaseModel) -> bool:
    field = type(value).model_fields.get("source_ids")
    if field is None:
        return False
    raw = getattr(value, "source_ids")  # noqa: B009 - field is optional on BaseModel
    return isinstance(raw, list)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
