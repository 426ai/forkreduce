"""Explicitly approved, request-budgeted five-model smoke; never the full grid.

Run availability before CUAD. Both phases share one persistent spending ledger.
All generated files (including public CUAD answers) stay in ignored results/.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, ToolOutput, models

from benchmarks.data import DEFAULT_SEED, load_contracts, sample_contracts, select_questions
from benchmarks.estimate import count_tokens, price_usage
from benchmarks.methods import METHODS, _code, _settings
from benchmarks.metrics import score, summarize_usage
from benchmarks.profiles import model_profile, output_mode_comparison, output_mode_decision, profile_output_mode
from benchmarks.providers import RecordingModel, provider_model
from benchmarks.reporting import summarize_rows, write_reports
from benchmarks.schema import FRAMEWORK_DEFAULTS, Answers, MethodConfig, MethodResult

# Official Standard/peak USD per million tokens; sources are persisted with results.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "deepseek-flash": {
        "api": "chat",
        "rates": (0.30, 0.006, 1.20),
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
        # Preserve measured tools runs; profile-derived prompted is a future paid-run decision.
        "method_configs": {
            "forkreduce": {"output_mode": "tools", "max_output_tokens": 8192},
            "one_shot": {"output_mode": "tools"},
        },
    },
    "gpt-6.1-sol": {
        "api": "responses",
        "rates": (2.0, 0.10, 10.0),
        "source": "https://developers.openai.com/api/docs/pricing",
    },
    "gpt-6-luna": {
        "api": "responses",
        "rates": (0.10, 0.01, 0.50),
        "requested_model": "gpt-6.1-luna",
        "source": "https://developers.openai.com/api/docs/pricing",
    },
    "grok-4.7": {"api": "chat", "rates": (2.0, 0.50, 6.0), "source": "https://docs.x.ai/developers/models"},
    "mimo-v2.6-pro": {
        "api": "chat",
        "rates": (0.435, 0.0036, 0.87),
        "source": "https://mimo.mi.com/docs/en-US/pricing",
        "method_configs": {
            "forkreduce": {"max_output_tokens": 8192},
            # Do not silently change the previously measured one-shot method.
            "one_shot": {"output_mode": "tools", "max_output_tokens": 8192},
        },
    },
}
CONFIG = MethodConfig(concurrency=1, max_output_tokens=2048)


def method_config(name: str, method: str, base: MethodConfig = CONFIG) -> MethodConfig:
    """Use profile capabilities unless the measured method has an explicit override."""
    mode = "tools" if method == "tool_agent" else profile_output_mode(model_profile(name))
    return replace(
        base,
        **{"output_mode": mode, **MODEL_SPECS[name].get("method_configs", {}).get(method, {})},
    )


def output_mode_metadata(name: str, method: str, config: MethodConfig, model: Any | None = None) -> dict[str, Any]:
    overrides = MODEL_SPECS[name].get("method_configs", {}).get(method, {})
    return {
        **(
            output_mode_comparison(model, _settings(config))
            if model is not None
            else output_mode_decision(model_profile(name))
        ),
        "output_mode": config.output_mode,
        "source": "method_configs"
        if "output_mode" in overrides
        else ("tool_agent" if method == "tool_agent" else "profile"),
    }


def rates_for(name: str) -> dict[str, Any]:
    miss, hit, output = MODEL_SPECS[name]["rates"]
    # GPT cache population is 1.25x normal input. The proxy may not split
    # population tokens from misses; conservatively price every miss as a write.
    charged_miss = miss * 1.25 if name.startswith("gpt-") else miss
    return {
        "input_miss": charged_miss,
        "input_hit": hit,
        "cache_write": charged_miss,
        "output": output,
        "cache_policy": "automatic",
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def latest_runs(path: Path) -> list[dict[str, Any]]:
    """Keep all attempts in JSONL, but never pool blocked attempts as model scores."""
    from benchmarks.reporting import ACCURACY_METRICS

    latest = {(r["model"], r["contract_id"], r["method"]): r for r in _rows(path)}
    for row in latest.values():
        status = row.setdefault("execution_status", "executed" if row["provider_usage"] else "not_executed")
        if status == "not_executed":
            row["metrics"] = {**row["metrics"], **dict.fromkeys(ACCURACY_METRICS), "per_question": []}
    return list(latest.values())


def _model(name: str, ledger: Any, *, sticky: bool = True, max_output_tokens: int = 2048) -> Any:
    from benchmarks.budget import BudgetModel

    settings: dict[str, Any] = {"timeout": 120}
    profile = model_profile(name)
    if MODEL_SPECS[name]["api"] == "responses":
        settings.update(openai_prompt_cache_key=f"cuad-smoke-{name}", openai_store=False)
    if name == "grok-4.7" and sticky:
        settings["extra_headers"] = {"x-grok-conv-id": "cuad-smoke-20261024"}
    model = provider_model("openai", name, api=MODEL_SPECS[name]["api"], settings=settings, profile=profile)
    return BudgetModel(
        model,
        ledger,
        rates=rates_for(name),
        max_output_tokens=max_output_tokens,
        label=name,
        # DeepSeek Chat does not explicitly promise a reasoning-inclusive cap;
        # reserve its official absolute 384K output ceiling instead.
        billable_output_bound=393216 if name == "deepseek-flash" else None,
        # xAI explicitly excludes reasoning AND function calls from its cap.
        # No further Grok request is admitted without an external hard limit.
        total_output_bounded=name != "grok-4.7",
    )


async def availability(directory: Path, ledger: Any, names: list[str]) -> None:
    path = directory / "availability.jsonl"
    previous = _rows(path)
    completed = {(r["model"], r["case"]) for r in previous if r["structured_output"]}
    # Synthetic public text, not contractual gold or customer data. Identical prefix
    # across repetitions; only the short final instruction varies.
    prefix = "".join(f"Public cache probe record {i:04d}: alpha beta gamma delta epsilon.\n" for i in range(520))
    prefix_tokens = count_tokens(prefix)
    for name in names:
        cases = [("short", "", True), ("long_1", prefix, True), ("long_2", prefix, True), ("long_3", prefix, True)]
        if name == "grok-4.7":
            cases = [
                ("short", "", False),
                ("baseline_1", prefix, False),
                ("baseline_2", prefix, False),
                ("sticky_1", prefix, True),
                ("sticky_2", prefix, True),
                ("sticky_3", prefix, True),
            ]
        for case, text, sticky in cases:
            if (name, case) in completed:
                continue
            result = MethodResult()
            model = RecordingModel(_model(name, ledger, sticky=sticky), result)
            agent = Agent(
                model,
                output_type=ToolOutput(Answers, name="Answers", max_retries=0),
                retries=0,
                model_settings={"max_tokens": 2048, "temperature": 0},
            )
            agent.instrument = False
            start = time.perf_counter()
            failure = None
            schema_ok = False
            try:
                response = await agent.run(
                    text + '\nReturn exactly one answer with question_id="probe", '
                    "empty spans and empty source_ids. Do not reproduce the records."
                )
                schema_ok = (
                    len(response.output.answers) == 1
                    and response.output.answers[0].question_id == "probe"
                    and not response.output.answers[0].spans
                )
            except Exception as exc:
                failure = _code(exc)[0]
            usage = summarize_usage(result.usage)
            row = {
                "model": name,
                "requested_model": MODEL_SPECS[name].get("requested_model", name),
                "attempt": 1 + sum(r["model"] == name and r["case"] == case for r in previous),
                "api": MODEL_SPECS[name]["api"],
                "framework_defaults": dict(FRAMEWORK_DEFAULTS),
                "output_mode_resolution": {
                    **output_mode_comparison(model, {"max_tokens": 2048, "temperature": 0}),
                    "output_mode": "tools",
                    "source": "availability_probe",
                },
                "case": case,
                "sticky_header": name == "grok-4.7" and sticky,
                "prefix_proxy_tokens": prefix_tokens if text else 0,
                "structured_output": schema_ok,
                "failure": failure,
                "usage": usage,
                "provider_usage": result.usage,
                "metadata": result.metadata,
                "cost_usd": price_usage(usage, rates_for(name)),
                "elapsed_seconds": time.perf_counter() - start,
            }
            _append(path, row)
            print(
                json.dumps({key: row[key] for key in ("model", "case", "structured_output", "failure", "cost_usd")}),
                flush=True,
            )


def selected_contracts(cache: Path) -> list[Any]:
    # Reuse prior 20-contract sample. Pick its two shortest substantial documents;
    # no gold labels influence selection. This is deliberately not a leaderboard.
    candidates = [
        c for c in sample_contracts(load_contracts(cache), 20, DEFAULT_SEED) if 2000 <= count_tokens(c.text) < 8000
    ]
    return sorted(candidates, key=lambda c: (count_tokens(c.text), c.id))[:2]


async def cuad(directory: Path, cache: Path, ledger: Any, names: list[str], *, n: int = 5) -> None:
    path = directory / "runs.jsonl"
    previous = _rows(path)
    completed = {(r["model"], r["contract_id"], r["method"]) for r in previous if r["provider_usage"]}
    contracts = selected_contracts(cache)
    configured_models = {
        name: _model(
            name,
            ledger,
            max_output_tokens=max(method_config(name, method).max_output_tokens for method in METHODS),
        )
        for name in MODEL_SPECS
    }
    manifest = {
        "seed": DEFAULT_SEED,
        "n": n,
        "config": asdict(CONFIG),
        "framework_defaults": dict(FRAMEWORK_DEFAULTS),
        "models": MODEL_SPECS,
        "resolved_methods": {
            name: {
                method: {
                    "config": asdict(method_config(name, method)),
                    "output_mode_resolution": output_mode_metadata(
                        name, method, method_config(name, method), configured_models[name]
                    ),
                }
                for method in METHODS
            }
            for name in MODEL_SPECS
        },
        "contracts": [
            {
                "id": c.id,
                "proxy_tokens": count_tokens(c.text),
                "sha256": hashlib.sha256(c.text.encode()).hexdigest(),
                "question_ids": [q.id for q in select_questions(c, n)],
            }
            for c in contracts
        ],
    }
    manifest = json.loads(json.dumps(manifest))
    manifest_path = directory / "workload.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("smoke_workload_changed")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for name in names:
        model = configured_models[name]
        jobs = [(contract, method) for contract in contracts for method in METHODS]
        random.Random(DEFAULT_SEED).shuffle(jobs)
        for contract, method in jobs:
            if (name, contract.id, method) in completed:
                continue
            questions = select_questions(contract, n)
            config = method_config(name, method)
            before = await ledger.snapshot()
            result = await METHODS[method](contract, questions, model, config)
            usage = summarize_usage(result.usage)
            after = await ledger.snapshot()
            sent = sum(r["status"] != "refused" for r in after["records"][len(before["records"]) :])
            row = {
                "model": name,
                "provider": "openai",
                "api": MODEL_SPECS[name]["api"],
                "contract_id": contract.id,
                "method": method,
                "n": n,
                "seed": DEFAULT_SEED,
                "attempt": 1
                + sum(
                    r["model"] == name and r["contract_id"] == contract.id and r["method"] == method for r in previous
                ),
                "sent_requests": sent,
                "execution_status": "executed" if sent else "not_executed",
                "config": asdict(config),
                "framework_defaults": dict(FRAMEWORK_DEFAULTS),
                "output_mode_resolution": output_mode_metadata(name, method, config, model),
                "metrics": score(contract, questions, result.answers),
                "usage": usage,
                "provider_usage": result.usage,
                "cost_usd": price_usage(usage, rates_for(name)),
                "elapsed_seconds": result.elapsed_seconds,
                "concurrency": result.concurrency,
                "schema_failures": result.schema_failures,
                "retries": result.retries,
                "failures": result.failures,
                "metadata": result.metadata,
                "answers": result.answers.model_dump(),
            }
            _append(path, row)
            print(json.dumps({key: row[key] for key in ("model", "method", "cost_usd", "failures")}), flush=True)
            # Write each model separately: the legacy reporter groups by method/N.
            model_rows = [r for r in latest_runs(path) if r["model"] == name]
            write_reports(model_rows, directory / name)
    all_rows = latest_runs(path)
    summaries = []
    for name in MODEL_SPECS:
        model_rows = [r for r in all_rows if r["model"] == name]
        if not model_rows:
            continue
        write_reports(model_rows, directory / name)
        for summary in summarize_rows(model_rows):
            selected = [r for r in model_rows if r["method"] == summary["method"]]
            summary["executed_records"] = sum(r["execution_status"] == "executed" for r in selected)
            summary["blocked_records"] = len(selected) - summary["executed_records"]
            summaries.append(summary)
    if summaries:
        with (directory / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        (directory / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


async def run(args: argparse.Namespace) -> None:
    if not args.approve_paid_smoke or os.environ.get("FORKREDUCE_BENCH_ALLOW_PAID") != "1":
        raise ValueError("paid_execution_disabled")
    if not math.isfinite(args.budget_usd) or not 0 < args.budget_usd <= 3.0:
        raise ValueError("smoke_budget_must_be_positive_and_at_most_three_usd")
    if not 1 <= args.n <= 41:
        raise ValueError("smoke_question_count_must_be_between_one_and_41")
    # Secrets are read only at runtime and never persisted in any manifest.
    if not os.environ.get("OPENAI_BASE_URL"):
        raise ValueError("smoke_base_url_missing")
    key = os.environ.get("PROXY_API_KEY")
    if not key:
        key = Path("/home/box/.config/proxy_key").read_text().strip()
    os.environ["OPENAI_API_KEY"] = key
    from benchmarks.budget import BudgetLedger

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(args.budget_usd, args.output_dir / "budget.json")
    await ledger.raise_uncertain_output_bound("deepseek-flash", 393216, rates_for("deepseek-flash"))
    if args.quarantine_bound_model:
        await ledger.quarantine_output_bound_model(args.quarantine_bound_model)
    if args.resume_after_correction:
        await ledger.resume_after_uncertain(args.resume_after_correction)
    with models.override_allow_model_requests(True):
        if args.phase in ("availability", "all"):
            await availability(args.output_dir, ledger, args.models)
        if args.phase in ("cuad", "all"):
            await cuad(args.output_dir, args.cache_dir, ledger, args.models, n=args.n)
    snapshot = await ledger.snapshot()
    print(json.dumps({k: snapshot[k] for k in ("charged_usd", "remaining_usd", "measured_usd", "halted")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("availability", "cuad", "all"))
    parser.add_argument("--approve-paid-smoke", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=3.0, help="Persistent ledger ceiling, at most $3")
    parser.add_argument("--n", type=int, default=5, help="CUAD questions per contract (1–41)")
    parser.add_argument(
        "--resume-after-correction",
        metavar="REASON_CODE",
        help="Acknowledge a corrected failed request; preserve its full reserved charge",
    )
    parser.add_argument(
        "--quarantine-bound-model",
        choices=list(MODEL_SPECS),
        help="Permanently block an output-bound-violating model; preserve all charges",
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/models-smoke"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/cuad"))
    args = parser.parse_args()
    models.ALLOW_MODEL_REQUESTS = False
    try:
        asyncio.run(run(args))
    except Exception as exc:
        # Never expose provider bodies, URLs, secret-bearing settings, or tracebacks.
        print(f"Paid smoke stopped: {type(exc).__name__}", flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
