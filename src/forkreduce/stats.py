"""Usage and cache-hit aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.usage import RequestUsage, RunUsage

from forkreduce.query import QueryResult


def cache_hit_tokens(usage: RunUsage | RequestUsage) -> int:
    """Read provider cache hits.

    pydantic-ai 2.x exposes ``cache_read_tokens``. Some OpenAI-compatible
    providers instead report ``prompt_cache_hit_tokens`` inside ``usage.details``.
    When the first-class field is non-zero it wins, so the two are not added.
    """
    if usage.cache_read_tokens:
        return int(usage.cache_read_tokens)
    raw = usage.details.get("prompt_cache_hit_tokens", 0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    return int(raw)


@dataclass
class UsageSummary:
    """Tokens, cache hits, and elapsed time summed across calls."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    elapsed_ms: float = 0.0
    by_round: dict[int, UsageSummary] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        """``cache_hit_tokens / input_tokens``, or ``0.0`` when nothing was sent."""
        if self.input_tokens <= 0:
            return 0.0
        return self.cache_hit_tokens / self.input_tokens

    def add_result(self, result: QueryResult[object]) -> None:
        usage = result.usage
        self.requests += usage.requests
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_hit_tokens += result.cache_hit_tokens
        self.elapsed_ms += result.elapsed_ms
        bucket = self.by_round.get(result.round)
        if bucket is None:
            bucket = UsageSummary()
            self.by_round[result.round] = bucket
        bucket.requests += usage.requests
        bucket.input_tokens += usage.input_tokens
        bucket.output_tokens += usage.output_tokens
        bucket.cache_hit_tokens += result.cache_hit_tokens
        bucket.elapsed_ms += result.elapsed_ms


def summarize(results: list[QueryResult[object]]) -> UsageSummary:
    summary = UsageSummary()
    for result in results:
        summary.add_result(result)
    return summary


def add_usage(left: RunUsage, right: RunUsage) -> RunUsage:
    return left + right
