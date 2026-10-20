"""The offline example runs without a network or an API key."""

from __future__ import annotations

import pytest
from examples.run_offline import main


@pytest.mark.asyncio
async def test_offline_example() -> None:
    result = await main()
    assert result.failures == []
    assert len(result.rounds) == 2
    assert result.merged is not None
    assert result.merged.items
    assert result.cache_hit_rate > 0
    reasons = [reason for report in result.reports for reason in getattr(report, "reasons", [])]
    assert any(reason.item_id == "deadline" for reason in reasons)
    assert any(item.block_id == "delivery" for item in result.merged.items)
