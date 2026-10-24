"""Deterministic contract-cluster uncertainty and offline result artifacts."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

ACCURACY_METRICS = (
    "word_precision",
    "word_recall",
    "word_f1",
    "exact_match",
    "span_precision",
    "span_recall",
    "span_f1",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "citation_exact_match",
    "absence_accuracy",
    "answerable_f1",
)
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "cache_write_tokens",
)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values: Iterable[float | None]) -> float | None:
    values = [value for value in values if _number(value)]
    return sum(values) / len(values) if values else None


def _quantile(values: Sequence[float], fraction: float) -> float:
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def _cluster_interval(clusters: Mapping[str, Sequence[float]], *, samples: int, seed: int) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    keys = sorted(clusters)
    values = [value for key in keys for value in clusters[key]]
    if not values:
        return {"mean": None, "low": None, "high": None, "contracts": 0, "samples": samples}
    rng = random.Random(seed)
    estimates = []
    totals = {key: (sum(clusters[key]), len(clusters[key])) for key in keys}
    for _ in range(samples):
        total = count = 0
        for _ in keys:
            subtotal, size = totals[keys[rng.randrange(len(keys))]]
            total += subtotal
            count += size
        estimates.append(total / count)
    estimates.sort()
    return {
        "mean": sum(values) / len(values),
        "low": _quantile(estimates, 0.025),
        "high": _quantile(estimates, 0.975),
        "contracts": len(keys),
        "samples": samples,
    }


def bootstrap_ci(
    records: Iterable[Mapping[str, Any]], *, metric: str = "word_f1", samples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Percentile 95% CI resampling whole contract clusters, not questions or N."""
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in records:
        value = row.get("metrics", {}).get(metric)
        if _number(value):
            clusters[str(row["contract_id"])].append(float(value))
    return _cluster_interval(clusters, samples=samples, seed=seed)


def paired_comparison(
    rows: Iterable[Mapping[str, Any]],
    method_a: str,
    method_b: str,
    *,
    metric: str = "word_f1",
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare A minus B with exact (contract_id, N) pairing and cluster CI.

    Missing rows, duplicate pairs and unknown/nonfinite metrics are errors, not
    silently dropped observations. All N values of a contract resample together.
    """
    if method_a == method_b:
        raise ValueError("Paired comparison requires two distinct methods")
    selected: dict[str, dict[tuple[str, Any], Mapping[str, Any]]] = {method_a: {}, method_b: {}}
    for row in rows:
        method = row["method"]
        if method not in selected:
            continue
        key = (str(row["contract_id"]), row["n"])
        if key in selected[method]:
            raise ValueError(f"Duplicate (contract_id, n) pair for {method}")
        selected[method][key] = row
    left, right = selected[method_a], selected[method_b]
    if not left or set(left) != set(right):
        raise ValueError("Paired comparison requires identical nonempty (contract_id, n) sets")
    clusters: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left, key=lambda value: (value[0], str(value[1]))):
        a, b = left[key].get("metrics", {}).get(metric), right[key].get("metrics", {}).get(metric)
        if not _number(a) or not _number(b):
            raise ValueError("Paired comparison requires a known finite metric for every pair")
        clusters[key[0]].append(float(a) - float(b))
    interval = _cluster_interval(clusters, samples=bootstrap_samples, seed=seed)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "difference": interval["mean"],
        "ci_low": interval["low"],
        "ci_high": interval["high"],
        "pairs": len(left),
        "contracts": interval["contracts"],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "confidence": 0.95,
        "direction": "method_a_minus_method_b",
    }


def _sum_known(values: Iterable[Any]) -> tuple[float | int | None, int]:
    values = list(values)
    unknown = sum(not _number(value) for value in values)
    return (sum(values) if values and not unknown else None, unknown)


def _failure_count(row: Mapping[str, Any]) -> int:
    failures = row.get("failures", [])
    return len(failures) if isinstance(failures, (list, tuple)) else int(failures or 0)


def summarize_rows(
    rows: Iterable[Mapping[str, Any]], *, bootstrap_samples: int = 2000, seed: int = 0
) -> list[dict[str, Any]]:
    """Aggregate actual JSONL records by method and question count."""
    groups: dict[tuple[str, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["n"])].append(row)
    summaries = []
    for (method, n), records in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        result: dict[str, Any] = {
            "method": method,
            "n": n,
            "records": len(records),
            "contracts": len({row["contract_id"] for row in records}),
        }
        for metadata in ("model", "provider"):
            result[metadata] = ", ".join(
                sorted({str(row[metadata]) for row in records if row.get(metadata) is not None})
            )
        for metric in ACCURACY_METRICS:
            interval = bootstrap_ci(records, metric=metric, samples=bootstrap_samples, seed=seed)
            result[metric] = interval["mean"]
            result[f"{metric}_ci_low"] = interval["low"]
            result[f"{metric}_ci_high"] = interval["high"]
        for field in TOKEN_FIELDS:
            result[field], result[f"{field}_unknown_records"] = _sum_known(
                row.get("usage", {}).get(field) for row in records
            )
        result["cache_hit_rate"] = (
            result["cache_hit_tokens"] / result["input_tokens"]
            if result["cache_hit_tokens"] is not None and result["input_tokens"]
            else None
        )
        for field in ("cost_usd", "elapsed_seconds"):
            result[field], result[f"{field}_unknown_records"] = _sum_known(row.get(field) for row in records)
        result["known_cost_usd"] = sum(row["cost_usd"] for row in records if _number(row.get("cost_usd")))
        result["mean_elapsed_seconds"] = _mean(
            row["elapsed_seconds"] for row in records if _number(row.get("elapsed_seconds"))
        )
        result["concurrency_min"] = min(
            (row["concurrency"] for row in records if _number(row.get("concurrency"))), default=None
        )
        result["concurrency_max"] = max(
            (row["concurrency"] for row in records if _number(row.get("concurrency"))), default=None
        )
        result["schema_failures"] = sum(row.get("schema_failures", 0) for row in records)
        result["retries"] = sum(row.get("retries", 0) for row in records)
        result["failures"] = sum(_failure_count(row) for row in records)
        result["failed_records"] = sum(bool(_failure_count(row)) for row in records)
        result["usage_missing_records"] = sum(bool(row.get("usage", {}).get("usage_missing", True)) for row in records)
        for field in (
            "literal_citations",
            "nonliteral_citations",
            "omitted_annotations",
            "missing_answers",
            "duplicate_answers",
            "unexpected_answers",
            "answerable_questions",
            "absence_questions",
            "gold_nonliteral_spans",
            "gold_omitted_spans",
            "gold_out_of_bounds",
            "citation_scored_questions",
            "invalid_gold_questions",
        ):
            result[field] = sum(row.get("metrics", {}).get(field, 0) for row in records)
        result["bootstrap_samples"] = bootstrap_samples
        result["seed"] = seed
        summaries.append(result)
    return summaries


def _positions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, Any, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for question in row.get("metrics", {}).get("per_question", []):
            groups[(row["method"], row["n"], question["sequence_position"])].append(question)
    return [
        {
            "method": method,
            "n": n,
            "sequence_position": position,
            "questions": len(items),
            "word_f1": _mean(item["word_f1"] for item in items),
            "exact_match": _mean(item["exact_match"] for item in items),
            "citation_f1": _mean(item["citation_f1"] for item in items),
        }
        for (method, n, position), items in sorted(
            groups.items(), key=lambda item: (item[0][0], str(item[0][1]), item[0][2])
        )
    ]


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], default_fields: Sequence[str]) -> None:
    fields = list(rows[0]) if rows else list(default_fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _display(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[str]:
    return [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *["| " + " | ".join(_display(row.get(column)) for column in columns) + " |" for row in rows],
    ]


def _svg(summaries: Sequence[Mapping[str, Any]], positions: Sequence[Mapping[str, Any]]) -> str:
    panels = [
        ("word_f1", "Word F1 / contract-cluster 95% CI"),
        ("citation_f1", "Citation character F1 / 95% CI"),
        ("input_tokens", "Provider input tokens (inclusive)"),
        ("output_tokens", "Provider output tokens"),
        ("cache_hit_rate", "Observed cache hit rate"),
        ("elapsed_seconds", "Summed elapsed seconds"),
        ("failures", "Failures"),
        ("cost_usd", "Cost USD (unknown is not zero)"),
    ]
    width = 960
    row_height = max(110, 52 + len(summaries) * 23)
    height = 45 + row_height * (len(panels) // 2) + max(80, len(positions) * 23 + 45)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<g font-family="sans-serif" font-size="12" fill="currentColor">',
        '<text x="20" y="24" font-size="18">CUAD measured results</text>',
    ]
    if not summaries:
        parts.append('<text x="20" y="52">No result records.</text>')
    for index, (metric, title) in enumerate(panels):
        x, y = 20 + (index % 2) * 470, 50 + (index // 2) * row_height
        parts.append(f'<text x="{x}" y="{y}" font-weight="bold">{escape(title)}</text>')
        known = [row[metric] for row in summaries if _number(row.get(metric))]
        scale = 1.0 if metric.endswith("f1") or metric.endswith("rate") else max(known, default=1) or 1
        for offset, row in enumerate(summaries):
            line_y = y + 23 * (offset + 1)
            label = f"{row['method']} N={row['n']}"
            value = row.get(metric)
            parts.append(f'<text x="{x}" y="{line_y}">{escape(label)}</text>')
            if _number(value):
                length = max(0.0, float(value)) / scale * 150
                parts.append(
                    f'<rect x="{x + 170}" y="{line_y - 11}" width="{length:.2f}" height="13" '
                    'fill="currentColor" fill-opacity="0.6"/>'
                )
                low, high = row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
                if _number(low) and _number(high):
                    parts.append(
                        f'<line x1="{x + 170 + low / scale * 150:.2f}" '
                        f'x2="{x + 170 + high / scale * 150:.2f}" y1="{line_y - 5}" y2="{line_y - 5}" '
                        'stroke="currentColor" stroke-width="2"/>'
                    )
            parts.append(f'<text x="{x + 330}" y="{line_y}">{escape(_display(value))}</text>')
    y = 50 + row_height * (len(panels) // 2)
    parts.append(
        f'<text x="20" y="{y}" font-weight="bold">'
        "Question sequence position: measured word F1 (including tool_agent)</text>"
    )
    for index, row in enumerate(positions):
        line_y = y + 23 * (index + 1)
        label = f"{row['method']} N={row['n']} question #{row['sequence_position']} ({row['questions']} records)"
        parts.append(f'<text x="20" y="{line_y}">{escape(label)}</text>')
        value = row["word_f1"]
        if _number(value):
            parts.append(
                f'<rect x="440" y="{line_y - 11}" width="{value * 300:.2f}" height="13" '
                'fill="currentColor" fill-opacity="0.6"/>'
            )
        parts.append(f'<text x="760" y="{line_y}">{escape(_display(value))}</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def write_reports(
    rows: Iterable[Mapping[str, Any]], output_dir: str | Path, *, bootstrap_samples: int = 2000, seed: int = 0
) -> dict[str, str]:
    """Write Markdown, flat CSV, position CSV and a dependency-free measured SVG."""
    rows = list(rows)
    summaries = summarize_rows(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    positions = _positions(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_dir / filename
        for name, filename in (
            ("markdown", "summary.md"),
            ("csv", "summary.csv"),
            ("svg", "summary.svg"),
            ("position_csv", "position.csv"),
        )
    }
    _csv(paths["csv"], summaries, ("method", "n", "records"))
    _csv(
        paths["position_csv"],
        positions,
        ("method", "n", "sequence_position", "questions", "word_f1", "exact_match", "citation_f1"),
    )
    text = [
        "# CUAD benchmark results",
        "",
        "All values below come from result records. Unknown usage or cost is not zero. "
        "Input tokens are cache-inclusive; cache writes are billed separately from hits and misses. "
        "Elapsed time is summed record wall time, not inferred suite wall time.",
        "",
        "## Accuracy",
        "",
        *_markdown_table(
            summaries,
            (
                "method",
                "n",
                "contracts",
                "word_f1",
                "word_f1_ci_low",
                "word_f1_ci_high",
                "exact_match",
                "span_f1",
                "absence_accuracy",
                "answerable_f1",
            ),
        ),
        "",
        f"95% percentile intervals resample whole contracts ({bootstrap_samples} bootstrap samples; seed {seed}). "
        "All N values of a contract remain in the same cluster. Gold spans are cumulative positives. "
        "Clause matching is one-to-one at word F1 >= 0.5.",
        "",
        "## Citations",
        "",
        *_markdown_table(
            summaries,
            (
                "method",
                "n",
                "citation_precision",
                "citation_recall",
                "citation_f1",
                "citation_f1_ci_low",
                "citation_f1_ci_high",
                "citation_exact_match",
                "literal_citations",
                "nonliteral_citations",
                "omitted_annotations",
                "gold_nonliteral_spans",
                "gold_omitted_spans",
                "gold_out_of_bounds",
                "citation_scored_questions",
                "invalid_gold_questions",
            ),
        ),
        "",
        "Citation character coverage includes every literal occurrence. Nonliteral and <omitted> annotations "
        "are counted explicitly and do not receive literal citation credit. Gold spans must match their exact "
        "source offset. If any gold span is nonliteral, annotated or out of bounds, that question has unknown "
        "character P/R/F1 and is excluded from character averages; literal citation location rates remain independent. "
        "Word and clause scores still use all gold text.",
        "",
        "## Provider usage and cost",
        "",
        *_markdown_table(
            summaries,
            (
                "method",
                "n",
                "input_tokens",
                "cache_miss_tokens",
                "cache_hit_tokens",
                "cache_write_tokens",
                "output_tokens",
                "total_tokens",
                "cache_hit_rate",
                "usage_missing_records",
                "cost_usd",
                "known_cost_usd",
                "cost_usd_unknown_records",
            ),
        ),
        "",
        "## Runtime and failures",
        "",
        *_markdown_table(
            summaries,
            (
                "method",
                "n",
                "elapsed_seconds",
                "mean_elapsed_seconds",
                "concurrency_min",
                "concurrency_max",
                "schema_failures",
                "retries",
                "failures",
                "failed_records",
                "missing_answers",
                "duplicate_answers",
                "unexpected_answers",
            ),
        ),
        "",
        "## Question position",
        "",
        "Positions are the one-based order of requested questions, not clause locations in the document. "
        "Each N and method (including tool_agent) is kept separate; missing answers retain their requested position "
        "and receive zero. This exposes late-question degradation.",
        "",
        *_markdown_table(
            positions, ("method", "n", "sequence_position", "questions", "word_f1", "exact_match", "citation_f1")
        ),
        "",
    ]
    paths["markdown"].write_text("\n".join(text), encoding="utf-8")
    paths["svg"].write_text(_svg(summaries, positions), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}
