"""Opt-in two-contract CUAD diagnosis with a separate, persistent USD 0.60 cap.

Only MiMo and DeepSeek are admitted. No availability probes or full grid run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pydantic_ai import models
from pydantic_ai.profiles.openai import OpenAIModelProfile

from benchmarks.budget import BudgetHalted, BudgetLedger, BudgetModel
from benchmarks.data import select_questions
from benchmarks.estimate import price_usage
from benchmarks.live_smoke import MODEL_SPECS, _model, rates_for, selected_contracts
from benchmarks.methods import METHODS
from benchmarks.metrics import score, summarize_usage
from benchmarks.providers import provider_model
from benchmarks.schema import MethodConfig


async def run(args: argparse.Namespace) -> None:
    if not args.approve_paid_smoke or os.environ.get("FORKREDUCE_BENCH_ALLOW_PAID") != "1":
        raise ValueError("paid_execution_disabled")
    if not os.environ.get("OPENAI_BASE_URL"):
        raise ValueError("smoke_base_url_missing")
    key = os.environ.get("PROXY_API_KEY") or Path("/home/box/.config/proxy_key").read_text().strip()
    os.environ["OPENAI_API_KEY"] = key
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(0.60, args.output_dir / "budget.json")
    if args.resume_after_correction:
        await ledger.resume_after_uncertain(args.resume_after_correction)
    config = MethodConfig(
        concurrency=1,
        max_output_tokens=args.max_output_tokens,
        output_mode=args.output_mode,
        retries=args.retries,
        raw_response_path=str(args.output_dir / f"{args.label}-failures.jsonl") if args.capture_failures else None,
    )
    configs = {
        method: replace(
            config,
            **(MODEL_SPECS[args.model].get("method_configs", {}).get(method, {}) if args.recommended else {}),
        )
        for method in args.methods
    }
    if args.model == "deepseek-flash":
        if args.tool_choice_auto:
            raise ValueError("deepseek_comparison_uses_its_required_capability_profile")
        model = _model(
            args.model, ledger, max_output_tokens=max(current.max_output_tokens for current in configs.values())
        )
    else:
        profile = OpenAIModelProfile(supports_forced_tool_choice=False) if args.tool_choice_auto else None
        model = BudgetModel(
            provider_model("openai", args.model, settings={"timeout": 120}, profile=profile),
            ledger,
            rates=rates_for(args.model),
            max_output_tokens=max(current.max_output_tokens for current in configs.values()),
            label=args.model,
        )
    path = args.output_dir / "runs.jsonl"
    contracts = selected_contracts(args.cache_dir)
    if len(contracts) != 2:
        raise ValueError("two_smoke_contracts_required")
    with models.override_allow_model_requests(True):
        for contract in contracts:
            questions = select_questions(contract, 5)
            for method in args.methods:
                config = configs[method]
                before = await ledger.snapshot()
                # Leave headroom rather than attempting experiments at the cap.
                if before["halted"] or before["remaining_usd"] < 0.05:
                    raise BudgetHalted("diagnostic_budget_headroom_exhausted")
                result = await METHODS[method](contract, questions, model, config)
                after = await ledger.snapshot()
                usage = summarize_usage(result.usage)
                row: dict[str, Any] = {
                    "label": args.label,
                    "model": args.model,
                    "contract_id": contract.id,
                    "document_sha256": hashlib.sha256(contract.text.encode()).hexdigest(),
                    "question_ids": [q.id for q in questions],
                    "method": method,
                    "n": 5,
                    "config": asdict(config),
                    "provider_settings": {"timeout": 120, "tool_choice_auto": args.tool_choice_auto},
                    "recommended_preset": args.recommended,
                    "answers": result.answers.model_dump(),
                    "failures": result.failures,
                    "schema_failures": result.schema_failures,
                    "retries": result.retries,
                    "metadata": result.metadata,
                    "provider_usage": result.usage,
                    "metrics": score(contract, questions, result.answers),
                    "cost_usd": price_usage(usage, rates_for(args.model)),
                    "charged_delta_usd": after["charged_usd"] - before["charged_usd"],
                    "elapsed_seconds": result.elapsed_seconds,
                    "ledger_request_ids": [r["request_id"] for r in after["records"][len(before["records"]) :]],
                }
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "label",
                                "model",
                                "method",
                                "contract_id",
                                "failures",
                                "cost_usd",
                                "elapsed_seconds",
                            )
                        }
                    ),
                    flush=True,
                )
                if after["halted"]:
                    raise BudgetHalted("diagnostic_request_uncertain")
    snapshot = await ledger.snapshot()
    print(json.dumps({k: snapshot[k] for k in ("charged_usd", "remaining_usd", "halted")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve-paid-smoke", action="store_true")
    parser.add_argument("--model", choices=("mimo-v2.6-pro", "deepseek-flash"), default="mimo-v2.6-pro")
    parser.add_argument("--methods", nargs="+", choices=("forkreduce", "one_shot"), default=["forkreduce", "one_shot"])
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--recommended", action="store_true", help="Apply model/method mode and output-cap overrides from live_smoke"
    )
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--output-mode", choices=("tools", "prompted"), default="tools")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--tool-choice-auto", action="store_true")
    parser.add_argument("--capture-failures", action="store_true")
    parser.add_argument("--resume-after-correction", metavar="REASON_CODE")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/mimo-fix"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/cuad"))
    args = parser.parse_args()
    if not args.label or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.label
    ):
        parser.error("label must contain only letters, digits, underscores or hyphens")
    models.ALLOW_MODEL_REQUESTS = False
    try:
        asyncio.run(run(args))
    except Exception as exc:
        print(f"MiMo diagnosis stopped: {type(exc).__name__}", flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
