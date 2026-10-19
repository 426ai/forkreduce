"""Retry, concurrency, and budget policy.

Business rules (which missing category must be explained, which citation failure
is fatal) live in caller-supplied ``decide`` functions. This module only carries
the knobs the framework itself enforces, plus the action values those functions
return.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Accept:
    """Keep the current result. Uncovered or invalid ids stay on the report."""


@dataclass(frozen=True)
class RequireReason:
    """Continue the reduce branch and ask the model why each listed id was dropped.

    The continuation reuses the reduce message history, so the provider prefix
    cache still matches. ``items`` are input ids from the coverage report.
    """

    items: tuple[str, ...]


@dataclass(frozen=True)
class Retry:
    """Run the reduce again. ``items`` limits the next input; an empty tuple retries all.

    ``hint`` is appended to the reduce tail (not to the frozen instructions).
    """

    items: tuple[str, ...]
    hint: str


@dataclass(frozen=True)
class Fail:
    """Stop. ``reason`` is stored on the run and is not hidden."""

    reason: str


@dataclass(frozen=True)
class Drop:
    """Leave the listed items out of downstream inputs. They stay on the report.

    Used by citation-failure and gap-fill lead policies. Semantic reduce uses
    :class:`Accept`, :class:`RequireReason`, :class:`Retry`, and :class:`Fail`.
    """

    items: tuple[str, ...] = ()


Action = Accept | RequireReason | Retry | Fail | Drop


@dataclass
class Budget:
    """Hard ceilings checked before each model call.

    Token ceilings are cumulative across the session. When any token ceiling is
    set, calls are serialized, even if ``Policy.concurrency`` is greater than
    one. A call is refused only when completed calls have already reached the
    ceiling. The single in-flight call that crosses a ceiling is allowed,
    because its size is known only after it returns, and no other call starts
    until that usage is recorded. Overshoot is therefore at most one call.

    Request-count ceilings stay concurrent. Each call reserves one request
    before it is sent, so in-flight calls count. A call blocked before it is
    sent reports ``attempts=0`` and does not reserve a request.
    """

    max_requests: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_requests", "max_input_tokens", "max_output_tokens", "max_total_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass
class BudgetReport:
    """What the framework knows when a call would exceed the budget."""

    kind: str
    message: str
    requests: int
    input_tokens: int
    output_tokens: int


@dataclass
class Policy:
    """Execution limits for a session.

    ``repair_failure`` is the default action after citation repair is exhausted:

    * ``needs_review`` keeps the output and flags it
    * ``drop`` removes it from reduce inputs
    * ``fail`` aborts the run

    The output is never discarded silently; dropped and failing items stay on the
    run report.
    """

    concurrency: int = 8
    retries: int = 2
    retry_backoff: float = 0.5
    max_backoff: float = 8.0
    budget: Budget | None = None
    repair_failure: str = "needs_review"
    budget_policy: Callable[[BudgetReport], Action] | None = None
    context_limit: int | None = None
    context_reserve: int | None = None

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.retries < 0:
            raise ValueError("retries must be >= 0")
        if self.retry_backoff < 0 or self.max_backoff < 0:
            raise ValueError("backoff must be >= 0")
        if self.repair_failure not in {"needs_review", "drop", "fail"}:
            raise ValueError("repair_failure must be 'needs_review', 'drop', or 'fail'")
        if self.context_limit is not None and self.context_limit < 0:
            raise ValueError("context_limit must be >= 0")
        if self.context_reserve is not None and self.context_reserve < 0:
            raise ValueError("context_reserve must be >= 0")


def action_name(action: Action) -> str:
    return type(action).__name__


def describe(value: Any) -> str:
    return type(value).__name__
