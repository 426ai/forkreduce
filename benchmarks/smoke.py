"""Exercise all adapters, scoring, provider-shaped usage, and report files offline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic_ai.models import override_allow_model_requests

from benchmarks.schema import Contract, MethodConfig, Question, Span


async def smoke() -> int:
    from benchmarks.methods import METHODS, offline_model
    from benchmarks.metrics import score, summarize_usage
    from benchmarks.reporting import write_reports

    quote = "The agreement renews annually."
    contract = Contract(
        "synthetic-offline-not-cuad",
        quote + "\nEither party may terminate on thirty days notice.",
        (
            Question("renewal", "Which renewal terms apply?", (Span(quote, 0),)),
            Question("absent", "Is there an exclusivity clause?"),
        ),
    )
    rows = []
    with override_allow_model_requests(False):
        for name, method in METHODS.items():
            result = await method(contract, contract.questions, offline_model(), MethodConfig())
            metrics = score(contract, contract.questions, result.answers)
            usage = summarize_usage(result.usage)
            if result.failures or metrics["word_f1"] != 1.0 or not usage["input_tokens"]:
                raise RuntimeError(f"offline_smoke_failed_{name}")
            rows.append(
                {
                    "contract_id": contract.id,
                    "method": name,
                    "n": 2,
                    "metrics": metrics,
                    "usage": usage,
                    "cost_usd": 0.0,
                    "elapsed_seconds": result.elapsed_seconds,
                    "concurrency": result.concurrency,
                    "schema_failures": result.schema_failures,
                    "retries": result.retries,
                    "failures": result.failures,
                }
            )
    with tempfile.TemporaryDirectory(prefix="cuad-offline-") as directory:
        paths = write_reports(rows, Path(directory), bootstrap_samples=100)
        print(Path(paths["markdown"]).read_text())
    print("Offline smoke passed: three methods, answerable + absent clauses, no provider requests.")
    print("Synthetic usage validates accounting only; not CUAD accuracy or a measured provider cost.")
    return 0
