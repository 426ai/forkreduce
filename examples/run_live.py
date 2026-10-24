"""Optional live run against an OpenAI-compatible chat model.

Uses the same fictional notice as the offline example. Nothing runs unless
``OPENAI_API_KEY`` is set. ``OPENAI_BASE_URL`` and ``OPENAI_MODEL`` are optional;
``FORKREDUCE_MODEL`` is accepted as another name for the model.

    OPENAI_API_KEY=... OPENAI_MODEL=... python examples/run_live.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.notice import INSTRUCTIONS, NOTICE, Answer, FindingList  # noqa: E402
from forkreduce import CitationCheck, CitationRepair, Query, build, openai_compatible_model  # noqa: E402
from forkreduce.policy import Policy  # noqa: E402


async def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; skipping the live example.")
        return 0
    session = build(
        openai_compatible_model(),
        documents=NOTICE,
        instructions=INSTRUCTIONS,
        outputs=[Answer, FindingList],
        warmup=True,
        policy=Policy(concurrency=2, retries=1, retry_backoff=1.0),
    )
    result = await (
        session.map(
            [
                Query(tail="Which titles are eligible?", output=Answer, id="titles"),
                Query(tail="When must offers arrive?", output=Answer, id="deadline"),
            ]
        )
        .check(CitationCheck(), repair=CitationRepair(max_rounds=2))
        .reduce(output=FindingList, instructions="Merge the cited facts using the current input ids.")
        .run()
    )
    print(f"failures={len(result.failures)} cache_hit_rate={result.cache_hit_rate:.2f}")
    if result.merged is not None:
        print(result.merged.model_dump_json(indent=2))
    for failure in result.failures:
        print(f"{failure.kind}: {failure.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
