"""Source-tree CLI. Estimate and smoke never enable provider model requests."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import models  # noqa: E402

from benchmarks.schema import MethodConfig  # noqa: E402


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _config(args: argparse.Namespace) -> MethodConfig:
    path = args.method_config
    return MethodConfig(**json.loads(path.read_text())) if path else MethodConfig()


def _contracts(args: argparse.Namespace) -> list[Any]:
    from benchmarks.data import load_contracts, sample_contracts

    return sample_contracts(load_contracts(args.cache_dir), count=args.contracts, seed=args.seed)


def _workload(args: argparse.Namespace, contracts: list[Any]) -> dict[str, Any]:
    model_name = (
        os.environ.get("ANTHROPIC_MODEL")
        if args.provider == "anthropic"
        else os.environ.get("OPENAI_MODEL") or os.environ.get("FORKREDUCE_MODEL")
    )
    return {
        "seed": args.seed,
        "question_counts": args.n,
        "provider": args.provider,
        "model": model_name,
        "price_profile": args.price_profile,
        "methods": ["forkreduce", "tool_agent", "one_shot"],
        "method_config": asdict(_config(args)),
        "contracts": [{"id": contract.id, "sha256": _digest(contract.text.encode())} for contract in contracts],
    }


def estimate(args: argparse.Namespace) -> int:
    from benchmarks.estimate import EstimateConfig, estimate_grid, load_prices, render_markdown

    contracts = _contracts(args)
    assumptions = (
        EstimateConfig(**json.loads(args.estimate_config.read_text())) if args.estimate_config else EstimateConfig()
    )
    report = estimate_grid(
        contracts,
        load_prices(args.prices),
        question_counts=args.n,
        seed=args.seed,
        config=assumptions,
        method_config=_config(args),
    )
    report["workload"] = _workload(args, contracts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    content = _json(report).encode()
    path = args.output_dir / "estimate.json"
    path.write_bytes(content)
    markdown = render_markdown(report)
    (args.output_dir / "estimate.md").write_text(markdown)
    print(markdown)
    print(f"Estimate artifact: {path}; SHA256: {_digest(content)}")
    print("No provider calls. Paid execution requires separate user approval of this estimate.")
    return 0


def _approval(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("FORKREDUCE_BENCH_ALLOW_PAID") != "1":
        raise ValueError("paid_execution_disabled")
    if (
        not args.approve_estimate
        or not args.estimate_file
        or not args.budget_usd
        or not math.isfinite(args.budget_usd)
        or args.budget_usd <= 0
    ):
        raise ValueError("explicit_estimate_approval_and_positive_budget_required")
    content = args.estimate_file.read_bytes()
    if _digest(content) != args.approve_estimate:
        raise ValueError("approved_estimate_digest_mismatch")
    return json.loads(content)


async def run(args: argparse.Namespace) -> int:
    # Check authorization before downloading, constructing providers, or enabling requests.
    approval = _approval(args)
    contracts = _contracts(args)
    if approval.get("workload") != _workload(args, contracts):
        raise ValueError("approved_workload_mismatch")

    from benchmarks.data import select_questions
    from benchmarks.estimate import load_prices, price_usage
    from benchmarks.methods import METHODS
    from benchmarks.metrics import score, summarize_usage
    from benchmarks.providers import provider_model
    from benchmarks.reporting import write_reports

    prices = load_prices(args.prices)
    if prices != approval.get("prices"):
        raise ValueError("approved_prices_mismatch")
    price = prices["models"][args.price_profile]
    model = provider_model(args.provider)
    config = _config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_dir / "runs.jsonl"
    # Never mix/overwrite previously billed runs silently.
    if jsonl.exists():
        raise ValueError("output_runs_already_exist")
    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    rng = random.Random(args.seed)
    jobs = [(contract, n) for contract in contracts for n in args.n]
    rng.shuffle(jobs)
    with models.override_allow_model_requests(True), jsonl.open("x") as stream:
        for contract, n in jobs:
            order = list(METHODS)
            rng.shuffle(order)
            questions = select_questions(contract, n, seed=args.seed)
            for method in order:
                result = await METHODS[method](contract, questions, model, config)
                usage = summarize_usage(result.usage)
                cost = price_usage(usage, price)
                row = {
                    "contract_id": contract.id,
                    "contract_sha256": _digest(contract.text.encode()),
                    "method": method,
                    "n": n,
                    "seed": args.seed,
                    "provider": args.provider,
                    "model": model.model_name,
                    "price_profile": args.price_profile,
                    "approved_estimate_sha256": args.approve_estimate,
                    "config": asdict(config),
                    "versions": {
                        name: importlib.metadata.version(name) for name in ("forkreduce", "pydantic-ai", "tiktoken")
                    },
                    "metrics": score(contract, questions, result.answers),
                    "usage": usage,
                    "provider_usage": result.usage,
                    "cost_usd": cost,
                    "elapsed_seconds": result.elapsed_seconds,
                    "concurrency": result.concurrency,
                    "schema_failures": result.schema_failures,
                    "retries": result.retries,
                    "failures": result.failures,
                    "metadata": result.metadata,
                    "answers": result.answers.model_dump(),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
                print(f"{len(rows)} {method} N={n} cost_usd={cost}")
                # Post-response stop, not a provider-enforced hard spending cap.
                if cost is None or total_cost + cost >= args.budget_usd:
                    write_reports(rows, args.output_dir)
                    print("Stopped: unknown usage or post-response spending threshold reached.")
                    return 2
                total_cost += cost
    write_reports(rows, args.output_dir)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("estimate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--cache-dir", type=Path, default=Path(".cache/cuad"))
        command.add_argument("--contracts", type=int, default=20)
        command.add_argument("--seed", type=int, default=20261024)
        command.add_argument("--n", nargs="+", type=int, choices=(5, 20, 41), default=[5, 20, 41])
        command.add_argument("--prices", type=Path, default=Path("benchmarks/prices.json"))
        command.add_argument("--method-config", type=Path, help="JSON overrides for MethodConfig fields")
        command.add_argument("--estimate-config", type=Path, help="JSON overrides for EstimateConfig assumptions")
        command.add_argument("--output-dir", type=Path, default=Path("benchmarks/results") / name)
        command.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
        command.add_argument("--price-profile", default="deepseek-flash")
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--estimate-file", type=Path)
            command.add_argument("--approve-estimate", help="SHA256 of a separately user-approved estimate.json")
            command.add_argument("--budget-usd", type=float)
    commands.add_parser("smoke", help="Synthetic end-to-end FunctionModel run, no network or keys")
    summary = commands.add_parser("summarize")
    summary.add_argument("jsonl", type=Path)
    summary.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/summary"))
    return root


def main() -> int:
    args = parser().parse_args()
    models.ALLOW_MODEL_REQUESTS = False
    try:
        if args.command == "estimate" or (args.command == "run" and args.dry_run):
            return estimate(args)
        if args.command == "run":
            return asyncio.run(run(args))
        if args.command == "smoke":
            from benchmarks.smoke import smoke

            return asyncio.run(smoke())
        from benchmarks.reporting import write_reports

        rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
        print(_json(write_reports(rows, args.output_dir)))
        return 0
    except Exception as error:
        # SDK exceptions/URLs may contain credentials. Never print their repr or traceback.
        safe_codes = {
            "paid_execution_disabled",
            "explicit_estimate_approval_and_positive_budget_required",
            "approved_estimate_digest_mismatch",
            "approved_workload_mismatch",
            "approved_prices_mismatch",
            "output_runs_already_exist",
        }
        safe = str(error) if isinstance(error, ValueError) and str(error) in safe_codes else type(error).__name__
        print(f"Benchmark stopped: {safe}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
