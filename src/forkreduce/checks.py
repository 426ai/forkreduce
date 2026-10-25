"""Checks and repairs.

A check only reports. A repair builds a follow-up query, by default on the same
branch, and the framework copies only the allowed fields back onto the original
object. The model may return a full object; fields outside the allow-list are
ignored in code, not by asking the model nicely.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from forkreduce.documents import DocumentIndex
from forkreduce.errors import ForkReduceError
from forkreduce.query import CheckFailure, Query, QueryResult
from forkreduce.schemas import CITATION_INSTRUCTIONS
from forkreduce.textutil import normalize_whitespace

_PATH_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")


class Check(Protocol):
    """Report problems with one successful output. Do not call a model."""

    def __call__(self, output: Any, *, documents: DocumentIndex) -> list[CheckFailure]:
        """Return a failure for each problem. An empty list means the output passed."""
        ...


class Repair(Protocol):
    """Turn check failures into a follow-up query and a field-limited merge."""

    max_rounds: int

    def build_query(
        self,
        result: QueryResult[Any],
        failures: Sequence[CheckFailure],
        *,
        round_index: int,
    ) -> Query[Any]:
        """Build the repair query. Default implementations continue ``result.messages``."""
        ...

    def apply(
        self,
        original: Any,
        updated: Any,
        failures: Sequence[CheckFailure],
    ) -> Any:
        """Return ``original`` with only the allowed fields taken from ``updated``."""
        ...


@dataclass
class CitationCheck:
    """Locate each quote inside the block it cites.

    Matching ignores differences in whitespace. The quote and block field names
    default to ``quote`` and ``block_id``. Every nested model that has both
    fields is checked.
    Raw documents are not block-addressable and are explicitly rejected.
    """

    quote_field: str = "quote"
    block_field: str = "block_id"

    def __call__(self, output: Any, *, documents: DocumentIndex) -> list[CheckFailure]:
        if documents.document_format == "raw":
            raise ForkReduceError("CitationCheck requires document_format='blocks'; raw documents are not addressable")
        failures: list[CheckFailure] = []
        for path, node in _iter_citations(output, self.quote_field, self.block_field):
            quote = getattr(node, self.quote_field)
            block_id = getattr(node, self.block_field)
            quote_text = quote if isinstance(quote, str) else ""
            block_text_id = block_id if isinstance(block_id, str) else ""
            if not quote_text.strip():
                failures.append(
                    CheckFailure(
                        code="empty_quote",
                        message=f"empty quote at {path}",
                        path=path,
                        block_id=block_text_id or None,
                        quote=quote_text,
                    )
                )
                continue
            if not block_text_id:
                failures.append(
                    CheckFailure(
                        code="empty_block_id",
                        message=f"empty block_id at {path}",
                        path=path,
                        quote=quote_text,
                    )
                )
                continue
            source = documents.block_text(block_text_id)
            if source is None:
                failures.append(
                    CheckFailure(
                        code="unknown_block",
                        message=f"block {block_text_id!r} is not in the frozen documents",
                        path=path,
                        block_id=block_text_id,
                        quote=quote_text,
                    )
                )
                continue
            if normalize_whitespace(quote_text) not in normalize_whitespace(source):
                failures.append(
                    CheckFailure(
                        code="quote_not_found",
                        message=f"quote at {path} was not found in block {block_text_id}",
                        path=path,
                        block_id=block_text_id,
                        quote=quote_text,
                    )
                )
        return failures


@dataclass
class CitationRepair:
    """Repair citations by continuing the branch, then copying only allowed fields.

    ``max_rounds`` defaults to 2. The repair query asks for the same output type
    the original query used, which is already on the frozen tool set. After the
    model returns, :meth:`apply` copies ``allowed_fields`` and nothing else.
    """

    max_rounds: int = 2
    allowed_fields: tuple[str, ...] = ("quote", "block_id")

    def __post_init__(self) -> None:
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be >= 0")
        if not self.allowed_fields:
            raise ValueError("allowed_fields must not be empty")

    def build_query(
        self,
        result: QueryResult[Any],
        failures: Sequence[CheckFailure],
        *,
        round_index: int,
    ) -> Query[Any]:
        if result.output is None:
            raise ForkReduceError("cannot repair a query that has no output")
        output_type = type(result.output)
        lines = [
            "Some citations could not be located in the frozen documents.",
            "Return the same output type.",
            "You may change only these fields: " + ", ".join(self.allowed_fields) + ".",
            "Keep every other field identical to your previous answer.",
            CITATION_INSTRUCTIONS,
            "When only citation fields may change, shorten the quote to a passage inside one block "
            "or correct its block_id; do not change the output structure to split citations.",
            "",
            "Failures:",
        ]
        for failure in failures:
            lines.append(
                f"- path={failure.path} code={failure.code} "
                f"block_id={failure.block_id or ''} quote={failure.quote or ''}"
            )
        return Query(
            tail="\n".join(lines),
            output=output_type,
            id=f"{result.query_id}:repair:{round_index}",
            history=list(result.messages),
            meta={"repair_of": result.query_id, "repair_round": round_index},
            round=result.round,
        )

    def apply(
        self,
        original: Any,
        updated: Any,
        failures: Sequence[CheckFailure],
    ) -> Any:
        if not isinstance(original, BaseModel) or not isinstance(updated, BaseModel):
            raise ForkReduceError("citation repair requires pydantic models")
        if type(original) is not type(updated):
            raise ForkReduceError(f"repair returned {type(updated).__name__}, expected {type(original).__name__}")
        clone = original.model_copy(deep=True)
        for failure in failures:
            source = _resolve(updated, failure.path)
            target = _resolve(clone, failure.path)
            if not isinstance(source, BaseModel) or not isinstance(target, BaseModel):
                continue
            updates: dict[str, Any] = {}
            for name in self.allowed_fields:
                if name in type(target).model_fields and name in type(source).model_fields:
                    updates[name] = getattr(source, name)
            if not updates:
                continue
            replaced = target.model_copy(update=updates)
            clone = replaced if failure.path in {"", "$"} else _replace_at(clone, failure.path, replaced)
        return clone


@dataclass
class CheckReport:
    """What a check/repair pass did with one query. Kept even when the item is dropped."""

    query_id: str
    round: int
    failures: list[CheckFailure]
    repair_rounds: int
    action: str


def _iter_citations(
    value: Any,
    quote_field: str,
    block_field: str,
    prefix: str = "",
) -> list[tuple[str, BaseModel]]:
    found: list[tuple[str, BaseModel]] = []
    if isinstance(value, BaseModel):
        fields = type(value).model_fields
        path = prefix or "$"
        if quote_field in fields and block_field in fields:
            found.append((path, value))
        for name in fields:
            child = getattr(value, name)
            child_prefix = name if not prefix else f"{prefix}.{name}"
            found.extend(_iter_citations(child, quote_field, block_field, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_iter_citations(child, quote_field, block_field, f"{prefix}[{index}]"))
    return found


def _resolve(value: Any, path: str) -> Any:
    if path in {"", "$"}:
        return value
    current = value
    for match in _PATH_TOKEN.finditer(path):
        name, index = match.group(1), match.group(2)
        current = getattr(current, name) if name is not None else current[int(index)]
    return current


def _replace_at(root: Any, path: str, replacement: Any) -> Any:
    """Return ``root`` with ``replacement`` stored at ``path``."""
    tokens: list[str | int] = []
    for match in _PATH_TOKEN.finditer(path):
        name, index = match.group(1), match.group(2)
        tokens.append(name if name is not None else int(index))
    if not tokens:
        return replacement
    return _assign_tokens(root, tokens, replacement)


def _assign_tokens(current: Any, tokens: list[str | int], replacement: Any) -> Any:
    head, *rest = tokens
    if not rest:
        if isinstance(head, int):
            if not isinstance(current, list):
                raise ForkReduceError(f"expected a list while applying repair path, got {type(current)}")
            current[head] = replacement
            return current
        if not isinstance(current, BaseModel):
            raise ForkReduceError(f"expected a model while applying repair path, got {type(current)}")
        return current.model_copy(update={head: replacement})
    if isinstance(head, int):
        if not isinstance(current, list):
            raise ForkReduceError(f"expected a list while applying repair path, got {type(current)}")
        current[head] = _assign_tokens(current[head], rest, replacement)
        return current
    if not isinstance(current, BaseModel):
        raise ForkReduceError(f"expected a model while applying repair path, got {type(current)}")
    child = _assign_tokens(getattr(current, head), rest, replacement)
    return current.model_copy(update={head: child})
