"""Offline CUAD workload accounting; estimates, never measured model usage.

Token lengths use real cl100k_base encodings, a proxy for the two providers'
private tokenizers. No model clients or API credentials are used by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from benchmarks.data import CUAD_ARCHIVE_SHA256, CUAD_JSON_SHA256, CUAD_URL, DEFAULT_SEED, select_questions
from benchmarks.schema import INSTRUCTIONS, Answers, Contract, MethodConfig, Question, question_prompt

METHODS = ("forkreduce", "one_shot", "tool_agent")
TOKEN_FIELDS = ("input_tokens", "cache_miss_tokens", "cache_hit_tokens", "cache_write_tokens", "output_tokens")


@dataclass(frozen=True)
class EstimateConfig:
    """Explicit workload assumptions, configurable without making any model calls."""

    output_tokens_per_question: int = 256
    tool_loops_per_question: int = 2
    tool_call_output_tokens: int = 96
    tool_result_tokens: int | None = None
    tool_schema_tokens: int = 600
    request_overhead_tokens: int = 16
    repair_probability: float = 0.10
    repair_rounds: int = 2
    repair_instruction_tokens: int = 64

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if key == "tool_result_tokens" and value is None:
                continue
            if key != "repair_probability" and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{key} must be a nonnegative integer")
        if not math.isfinite(self.repair_probability) or not 0 <= self.repair_probability <= 1:
            raise ValueError("repair_probability must be between zero and one")


@lru_cache(maxsize=1)
def _encoding() -> Any:
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def token_length_distribution(contracts: Sequence[Contract]) -> dict[str, int | float | str]:
    """Actual encoded document lengths; percentiles use linear interpolation (type 7)."""
    lengths = sorted(count_tokens(contract.text) for contract in contracts)
    if not lengths:
        raise ValueError("At least one contract is required")

    def percentile(fraction: float) -> float:
        index = (len(lengths) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        return lengths[lower] + (lengths[upper] - lengths[lower]) * (index - lower)

    return {
        "tokenizer": "cl100k_base",
        "count": len(lengths),
        "min": lengths[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": lengths[-1],
        "mean": mean(lengths),
    }


def load_prices(path: Path | None = None) -> dict[str, Any]:
    """Load default or custom JSON, all four rates expressed in USD per million tokens."""
    path = Path(path) if path is not None else Path(__file__).with_name("prices.json")
    prices = json.loads(path.read_text(encoding="utf-8"))
    _validate_prices(prices)
    return prices


def _validate_prices(prices: Mapping[str, Any]) -> None:
    if prices.get("currency") != "USD" or prices.get("unit") != "per_million_tokens":
        raise ValueError("Prices must declare currency USD and unit per_million_tokens")
    if not prices.get("models"):
        raise ValueError("Prices must contain at least one model")
    for name, rates in prices["models"].items():
        for key in ("input_miss", "input_hit", "cache_write", "output"):
            value = rates.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} price for {name}")
        if rates.get("cache_policy", "explicit") not in ("explicit", "automatic"):
            raise ValueError(f"Invalid cache_policy for {name}")


def price_usage(usage: Mapping[str, int | float | None], rates: Mapping[str, Any]) -> float | None:
    """Price four disjoint categories; unknown usage stays unknown, never zero-filled."""
    categories = (
        ("cache_miss_tokens", "input_miss"),
        ("cache_hit_tokens", "input_hit"),
        ("cache_write_tokens", "cache_write"),
        ("output_tokens", "output"),
    )
    if any(usage.get(field) is None for field, _ in categories):
        return None
    return sum(float(usage[field]) * rates[price] for field, price in categories) / 1_000_000


@dataclass(frozen=True)
class _Request:
    input_tokens: int
    output_tokens: int
    cacheable_tokens: int
    reuse_tokens: int
    repairable: bool = True


def _workload(
    rendered_tokens: int,
    questions: tuple[Question, ...],
    method: str,
    config: EstimateConfig,
    section_count: int,
    tool_result_sizes: tuple[int, int],
) -> list[_Request]:
    common = count_tokens(INSTRUCTIONS) + count_tokens(json.dumps(Answers.model_json_schema(), sort_keys=True))
    static = common + rendered_tokens
    output = config.output_tokens_per_question
    overhead = config.request_overhead_tokens
    requests: list[_Request] = []
    question_tokens = count_tokens(question_prompt(questions))
    if method == "one_shot":
        requests.append(_Request(static + question_tokens + overhead, len(questions) * output, static, 0))
    elif method == "forkreduce":
        for index, question in enumerate(questions):
            suffix = count_tokens(question_prompt((question,))) + overhead
            requests.append(_Request(static + suffix, output, static, static if index else 0))
        # Map answers are paid output, then paid reduce input; reduce emits the final answers again.
        reduce_input = static + question_tokens + len(questions) * output + overhead
        requests.append(_Request(reduce_input, len(questions) * output, static, static))
    elif method == "tool_agent":
        static = common + config.tool_schema_tokens
        history = static
        previous_input = 0
        for question in questions:
            prompt = (
                question_prompt((question,)) + "\nUse search(query) and read_section(id,start,end) to inspect "
                "the contract as needed. Section ids are b1, b2, ... in document order. "
                f"The contract has {section_count} overlapping sections. "
                "Answer only this question; previous turns are retained context, not additional questions."
            )
            history += count_tokens(prompt)
            for step in range(config.tool_loops_per_question + 1):
                final = step == config.tool_loops_per_question
                generated = output if final else config.tool_call_output_tokens
                current = history + overhead
                requests.append(_Request(current, generated, current, previous_input, repairable=final))
                previous_input = current
                result_tokens = (
                    config.tool_result_tokens if config.tool_result_tokens is not None else tool_result_sizes[step % 2]
                )
                history = current + generated + (0 if final else result_tokens)
    else:
        raise ValueError(f"Unknown method: {method}")
    return requests


def _bill(
    requests: list[_Request],
    rates: Mapping[str, Any],
    mode: str,
    config: EstimateConfig,
) -> dict[str, Any]:
    usage: dict[str, Any] = {field: 0.0 for field in TOKEN_FIELDS}
    explicit = rates.get("cache_policy", "explicit") == "explicit"

    def add(request: _Request, weight: float = 1.0) -> None:
        reuse = request.reuse_tokens if mode == "warm" else 0
        new_cache = request.cacheable_tokens - reuse if mode == "warm" else 0
        write = new_cache if explicit else 0
        miss = request.input_tokens - reuse - write
        for field, value in (
            ("input_tokens", request.input_tokens),
            ("cache_miss_tokens", miss),
            ("cache_hit_tokens", reuse),
            ("cache_write_tokens", write),
            ("output_tokens", request.output_tokens),
        ):
            usage[field] += weight * value

    for request in requests:
        add(request)
    # Every structured answer may need repair: N leaves plus reduce, N agent
    # answers, or the single one-shot output. Intermediate tool calls are excluded.
    expected_repairs = 0.0
    repairable_count = 0
    for original in requests:
        if not original.repairable:
            continue
        repairable_count += 1
        repair_input = original.input_tokens
        previous_input = original.cacheable_tokens
        for attempt in range(1, config.repair_rounds + 1):
            probability = config.repair_probability**attempt
            expected_repairs += probability
            repair_input += original.output_tokens + config.repair_instruction_tokens + config.request_overhead_tokens
            request = _Request(repair_input, original.output_tokens, repair_input, previous_input)
            add(request, probability)
            previous_input = repair_input
    usage["requests"] = len(requests) + expected_repairs
    usage["base_requests"] = len(requests)
    usage["expected_repair_requests"] = expected_repairs
    usage["maximum_requests"] = len(requests) + repairable_count * config.repair_rounds
    usage["usd"] = price_usage(usage, rates)
    return usage


def estimate_grid(
    contracts: Sequence[Contract],
    prices: dict[str, Any] | None = None,
    *,
    question_counts: Sequence[int] = (5, 20, 41),
    models: Sequence[str] | None = None,
    methods: Sequence[str] = METHODS,
    cache_modes: Sequence[str] = ("cold", "warm"),
    config: EstimateConfig | None = None,
    seed: int = DEFAULT_SEED,
    method_config: MethodConfig | None = None,
) -> dict[str, Any]:
    """Estimate exactly the supplied grid (default 20 contracts yields 180 workload rows).

    Each row contains all model/cache scenarios, so neither prices nor cache modes
    multiply the logical workload count. Pass already sampled contracts to this API.
    """
    config = config or EstimateConfig()
    method_config = method_config or MethodConfig()
    from benchmarks.methods import chunk_document
    from forkreduce.documents import coerce_documents

    prices = load_prices() if prices is None else prices
    _validate_prices(prices)
    models = tuple(prices["models"]) if models is None else tuple(models)
    if not contracts or not question_counts or not methods or not models or not cache_modes:
        raise ValueError("Contracts and all grid dimensions must be nonempty")
    if len({contract.id for contract in contracts}) != len(contracts):
        raise ValueError("Contract ids must be unique")
    if any(mode not in ("cold", "warm") for mode in cache_modes):
        raise ValueError("Cache modes must be cold or warm")
    if any(model not in prices["models"] for model in models):
        raise ValueError("Every selected model must have a price entry")
    if any(method not in METHODS for method in methods):
        raise ValueError("Methods must be forkreduce, one_shot, or tool_agent")
    if any(n <= 0 for n in question_counts):
        raise ValueError("Question counts must be positive")
    rows: list[dict[str, Any]] = []
    snapshot = []
    for contract in contracts:
        document_tokens = count_tokens(contract.text)
        documents = coerce_documents(chunk_document(contract, method_config))
        blocks = list(documents.blocks.items())
        rendered_tokens = count_tokens(documents.rendered)
        search_result = [
            {"id": block_id, "start": int(block.meta["start"]), "length": len(block.text), "excerpt": block.text[:800]}
            for block_id, block in blocks[: method_config.search_results]
        ]
        first_id, first_block = blocks[0]
        read_end = min(len(first_block.text), method_config.read_chars)
        read_result = {
            "id": first_id,
            "start": 0,
            "end": read_end,
            "text": first_block.text[:read_end],
            "length": len(first_block.text),
        }
        tool_result_sizes = (count_tokens(json.dumps(search_result)), count_tokens(json.dumps(read_result)))
        snapshot.append(
            {
                "id": contract.id,
                "text_sha256": hashlib.sha256(contract.text.encode()).hexdigest(),
                "document_tokens": document_tokens,
                "rendered_document_tokens": rendered_tokens,
                "section_count": len(blocks),
                "modeled_tool_result_tokens": list(tool_result_sizes),
                "questions_sha256": hashlib.sha256(
                    json.dumps([{"id": q.id, "text": q.text} for q in contract.questions], sort_keys=True).encode()
                ).hexdigest(),
            }
        )
        for n in question_counts:
            questions = select_questions(contract, n, seed)
            for method in methods:
                requests = _workload(rendered_tokens, questions, method, config, len(blocks), tool_result_sizes)
                scenarios = {
                    model: {mode: _bill(requests, prices["models"][model], mode, config) for mode in cache_modes}
                    for model in models
                }
                rows.append(
                    {
                        "contract_id": contract.id,
                        "n": n,
                        "method": method,
                        "document_tokens": document_tokens,
                        "question_ids": [q.id for q in questions],
                        "scenarios": scenarios,
                    }
                )
    totals = []
    for n in question_counts:
        for method in methods:
            selected = [row for row in rows if row["n"] == n and row["method"] == method]
            for model in models:
                for mode in cache_modes:
                    usages = [row["scenarios"][model][mode] for row in selected]
                    totals.append(
                        {
                            "n": n,
                            "method": method,
                            "model": model,
                            "cache_mode": mode,
                            "contracts": len(selected),
                            **{field: sum(usage[field] for usage in usages) for field in usages[0]},
                        }
                    )
    dataset = {
        "source": CUAD_URL,
        "archive_sha256": CUAD_ARCHIVE_SHA256,
        "json_sha256": CUAD_JSON_SHA256,
        "contract_count": len(contracts),
        "contracts": snapshot,
        "seed": seed,
        "question_counts": list(question_counts),
    }
    dataset["input_snapshot_sha256"] = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()
    return {
        "kind": "offline_estimate",
        "workload_count": len(rows),
        "tokenizer": "cl100k_base",
        "distribution": token_length_distribution(contracts),
        "dataset": dataset,
        "config": asdict(config),
        "method_config": asdict(method_config),
        "prices": prices,
        "rows": rows,
        "totals": totals,
        "assumptions": [
            "Real cl100k_base document/question/schema lengths are a tokenizer proxy, not provider-billed usage; "
            "provider framing is modeled by request_overhead_tokens.",
            "Every method uses identical INSTRUCTIONS, canonical questions, Answers schema, and source contract. "
            "Full-context methods receive actual overlapping rendered sections; tool_agent initially receives only "
            "section metadata and obtains source text through search/read results, not the whole contract.",
            "Cold means cache unavailable: every request, including all growing agent history, is billed as input "
            "miss with zero hits/writes. Warm means an initially empty usable cache; its first population is billed "
            "and there is no free prewarming or cross-contract reuse.",
            "Equal exact-prefix cache eligibility applies to every method: static prefixes can hit after the first "
            "request and growing agent/repair histories reuse the complete previously processed input prefix. "
            "No suffix or changed-prefix hits are assumed.",
            "Warm fork accounting assumes one leaf establishes the common prefix before other leaves reuse it; "
            "concurrent cold cache-write races can cost more. Cold is the no-cache upper sensitivity. "
            "Cache TTL expiry, provider minimum prefix sizes and rounding are not modeled.",
            "Explicit cache-write tokens replace base input tokens; automatic cache population is billed as miss. "
            "Hit, miss, and write are disjoint and sum to input tokens; output is billed separately.",
            "Forkreduce uses N leaf requests plus one reduce request: leaf answers incur output charges, "
            "are read again as reduce input, and are emitted again as final output (double answer output).",
            "Tool agent processes questions sequentially with configured tool loops. Every request reprocesses "
            "the entire growing conversation, including previous questions, assistant tool/answer outputs, and "
            "tool results; eligible history prefix reads are still billed as cache hits.",
            "Tool loop result sizes, unless overridden, are actual cl100k encodings of representative search JSON "
            "(first search_results sections, 800-character excerpts) and first-section read JSON "
            "bounded by read_chars. "
            "Loops alternate search/read. Actual queries, sections, bounds and loop counts can differ; "
            "no retrieval accuracy is assumed.",
            "Every answer-bearing response reserves the same repair probability and rounds: N forks plus reduce, "
            "N agent answers, or one-shot. Expected repairs per response are sum(p**i for i in 1..repair_rounds). "
            "Failed answers and repair instructions are reprocessed. Compounded repair history in later agent "
            "turns and repair tool loops are not modeled; this is a planning estimate, not a worst-case cap.",
            "Prices are USD per million tokens. DeepSeek rates are peak; Sonnet cache writes use five-minute TTL. "
            "No discounts, taxes, reasoning-token overhead, network latency, or retries beyond the repair contingency "
            "are included.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render every per-N/method/model/cache aggregate, plus reproducible assumptions."""
    distribution = report["distribution"]
    lines = [
        "# CUAD offline cost estimate",
        "",
        f"Logical workloads: **{report['workload_count']}**; contracts: **{report['dataset']['contract_count']}**. "
        "No model calls were made.",
        f"Input snapshot SHA256: `{report['dataset']['input_snapshot_sha256']}`",
        "",
        "## Actual document token lengths (cl100k_base)",
        "",
        "| min | p25 | median | p75 | p95 | max | mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(f"{distribution[key]:,.2f}" for key in ("min", "p25", "median", "p75", "p95", "max", "mean"))
        + " |",
        "",
        "## Aggregate estimates",
        "",
        "All tokens/requests include the expected repair contingency. "
        "Cold and warm are alternative scenarios, not additive costs.",
        "",
        "| N | Method | Model | Cache | Contracts | Input | Miss | Hit | Write | Output | Requests | USD |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["totals"]:
        lines.append(
            f"| {row['n']} | {row['method']} | {row['model']} | {row['cache_mode']} | {row['contracts']} | "
            + " | ".join(
                f"{row[key]:,.2f}"
                for key in (
                    *TOKEN_FIELDS[:1],
                    "cache_miss_tokens",
                    "cache_hit_tokens",
                    "cache_write_tokens",
                    "output_tokens",
                    "requests",
                )
            )
            + f" | ${row['usd']:.6f} |"
        )
    lines.extend(["", "## Assumptions", ""])
    lines.extend(f"- {assumption}" for assumption in report["assumptions"])
    lines.extend(
        [
            "",
            "### Configured workload parameters",
            "",
            "```json",
            json.dumps(report["config"], indent=2),
            "```",
            "",
            "### Price sources",
            "",
        ]
    )
    lines.append(f"Verified date: {report['prices'].get('verified_at', 'custom / unspecified')}.")
    for model, rates in report["prices"]["models"].items():
        lines.append(
            f"- **{model}**: miss ${rates['input_miss']}, hit ${rates['input_hit']}, "
            f"write ${rates['cache_write']}, output ${rates['output']} / million tokens. "
            f"Source: {rates.get('source', 'custom JSON')}. {rates.get('notes', '')}"
        )
    return "\n".join(lines) + "\n"
