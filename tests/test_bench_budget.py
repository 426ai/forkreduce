"""Request budget and usage behavior, without credentials or network."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from benchmarks.budget import BudgetExceeded, BudgetHalted, BudgetLedger, BudgetModel, input_token_upper_bound
from benchmarks.providers import usage_record
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

RATES = {"input_miss": 1.0, "input_hit": 0.1, "cache_write": 1.25, "output": 2.0}
MESSAGES = [ModelRequest(parts=[UserPromptPart("private-contract-text")])]
PARAMETERS = ModelRequestParameters()


class StubModel(TestModel):
    """Return exactly the supplied usage, without FunctionModel's usage estimates."""

    def __init__(self, *, response: Any = None, error: Exception | None = None, block: bool = False) -> None:
        super().__init__()
        self.response = (
            response
            if response is not None
            else ModelResponse(parts=[], usage=RequestUsage(input_tokens=100, output_tokens=20, cache_read_tokens=10))
        )
        self.error = error
        self.calls_count = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.block = block

    async def request(self, messages: Any, model_settings: Any, model_request_parameters: Any) -> Any:
        self.calls_count += 1
        self.entered.set()
        if self.block:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.response


def wrap(model: StubModel, ledger: BudgetLedger, **kwargs: Any) -> BudgetModel:
    options = {"rates": RATES, "max_output_tokens": 100, "label": "synthetic/model"}
    options.update(kwargs)
    return BudgetModel(model, ledger, **options)


@pytest.mark.parametrize(
    "cache_details,reasoning_details",
    [("prompt_tokens_details", "completion_tokens_details"), ("input_tokens_details", "output_tokens_details")],
)
def test_both_apis_preserve_numeric_cached_and_reasoning_usage(cache_details: str, reasoning_details: str) -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            details={
                cache_details: {"cached_tokens": 75, "secret": "never-persist"},
                reasoning_details: {"reasoning_tokens": 12},
            },
        )
    )
    record = usage_record(response, stage="test")
    assert record["cache_read_tokens"] == 75
    assert record["reasoning_tokens"] == 12
    assert record["details"] == {"cached_tokens": 75, "reasoning_tokens": 12}
    assert "never-persist" not in repr(record)


@pytest.mark.parametrize("invalid", [True, -1, "12", float("nan"), float("inf")])
def test_reasoning_usage_rejects_noncounts(invalid: Any) -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            details={"reasoning_tokens": invalid, "output_tokens_details": {"reasoning_tokens": invalid}},
        )
    )
    record = usage_record(response, stage="test")
    assert record["reasoning_tokens"] is None
    assert "reasoning_tokens" not in record["details"]


async def test_budget_refuses_before_provider_request(tmp_path: Path) -> None:
    ledger = BudgetLedger(0.000001, tmp_path / "ledger.json")
    model = StubModel()
    with pytest.raises(BudgetExceeded, match="reservation_exceeds_remaining"):
        await wrap(model, ledger).request(MESSAGES, None, PARAMETERS)
    assert model.calls_count == 0
    state = await ledger.snapshot()
    assert state["charged_usd"] == 0
    assert state["reserved_usd"] == 0
    assert state["records"][0]["status"] == "refused"
    assert not state["halted"]


async def test_shared_concurrent_reservations_are_atomic_and_release_actual_cost(tmp_path: Path) -> None:
    bound = input_token_upper_bound(MESSAGES, {"max_tokens": 100, "max_retries": 0}, PARAMETERS)
    reservation = (bound * 1.25 + 100 * 2) / 1_000_000
    ledger = BudgetLedger(reservation * 1.5, tmp_path / "ledger.json")
    left, right = StubModel(block=True), StubModel()
    running = asyncio.create_task(wrap(left, ledger, label="left").request(MESSAGES, None, PARAMETERS))
    await left.entered.wait()
    try:
        state = await ledger.snapshot()
        assert state["reserved_usd"] >= reservation
        with pytest.raises(BudgetExceeded):
            await wrap(right, ledger, label="right").request(MESSAGES, None, PARAMETERS)
        assert right.calls_count == 0
    finally:
        left.release.set()
        await running
    await wrap(right, ledger, label="right").request(MESSAGES, None, PARAMETERS)
    assert right.calls_count == 1
    state = await ledger.snapshot()
    assert state["reserved_usd"] == 0
    assert state["charged_usd"] == pytest.approx(2 * (90 * 1 + 10 * 0.1 + 20 * 2) / 1_000_000)
    assert state["measured_usd"] == pytest.approx(state["charged_usd"])


@pytest.mark.parametrize("transport", [False, True])
async def test_missing_usage_or_transport_charges_reservation_and_halts(tmp_path: Path, transport: bool) -> None:
    ledger = BudgetLedger(3, tmp_path / "ledger.json")
    model = StubModel(
        response=ModelResponse(parts=[]),
        error=RuntimeError("private-provider-error-secret") if transport else None,
    )
    if transport:
        with pytest.raises(RuntimeError, match="private-provider-error-secret"):
            await wrap(model, ledger).request(MESSAGES, None, PARAMETERS)
    else:
        await wrap(model, ledger).request(MESSAGES, None, PARAMETERS)
    state = await ledger.snapshot()
    assert state["halted"]
    assert state["charged_usd"] == state["records"][0]["reservation_usd"]
    assert state["measured_usd"] is None
    assert state["reserved_usd"] == 0
    with pytest.raises(BudgetHalted):
        await wrap(model, ledger).request(MESSAGES, None, PARAMETERS)
    assert model.calls_count == 1
    saved = (tmp_path / "ledger.json").read_text()
    assert "private-contract-text" not in saved
    assert "private-provider-error-secret" not in saved
    assert json.loads(saved)["halted"]


async def test_known_counts_unknown_cache_settle_conservatively_without_halting(tmp_path: Path) -> None:
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=20, details={}))
    ledger = BudgetLedger(3, tmp_path / "ledger.json")
    model = wrap(StubModel(response=response), ledger)
    await model.request(MESSAGES, None, PARAMETERS)
    await model.request(MESSAGES, None, PARAMETERS)
    state = await ledger.snapshot()
    assert not state["halted"]
    assert state["measured_usd"] is None
    assert state["charged_usd"] == pytest.approx(2 * (100 * 1.25 + 20 * 2) / 1_000_000)
    assert state["records"][0]["usage"]["cache_read_tokens"] is None


async def test_unknown_official_price_requires_ceiling_and_never_claims_measured_cost(tmp_path: Path) -> None:
    ledger = BudgetLedger(3, tmp_path / "ledger.json")
    with pytest.raises(ValueError, match="reservation_rates_required"):
        wrap(StubModel(), ledger, rates=None)
    model = wrap(StubModel(), ledger, rates=None, reservation_rates=RATES)
    await model.request(MESSAGES, None, PARAMETERS)
    state = await ledger.snapshot()
    assert state["measured_usd"] is None
    assert state["charged_usd"] > 0
    assert not state["halted"]


async def test_ledger_resumes_allowance_and_refuses_limit_changes(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = BudgetLedger(3, path)
    await wrap(StubModel(), ledger).request(MESSAGES, None, PARAMETERS)
    before = await ledger.snapshot()
    reopened = BudgetLedger(3, path)
    after = await reopened.snapshot()
    assert after["charged_usd"] == before["charged_usd"]
    assert after["remaining_usd"] == before["remaining_usd"]
    with pytest.raises(ValueError, match="budget_limit_mismatch"):
        BudgetLedger(4, path)


async def test_interrupted_reservation_is_charged_and_halted_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = BudgetLedger(3, path)
    await ledger.reserve(label="interrupted", amount=Decimal("0.2"), input_bound=100, output_bound=20)
    resumed = BudgetLedger(3, path)
    state = await resumed.snapshot()
    assert state["charged_usd"] == 0.2
    assert state["reserved_usd"] == 0
    assert state["halt_reason"] == "interrupted_request"
    assert state["measured_usd"] is None
    with pytest.raises(BudgetHalted):
        await wrap(StubModel(), resumed).request(MESSAGES, None, PARAMETERS)


async def test_token_overrun_halts_even_when_price_below_reservation(tmp_path: Path) -> None:
    ledger = BudgetLedger(3, tmp_path / "ledger.json")
    model = StubModel(response=ModelResponse(parts=[], usage=RequestUsage(input_tokens=100, output_tokens=101)))
    await wrap(model, ledger).request(MESSAGES, None, PARAMETERS)
    state = await ledger.snapshot()
    assert state["halt_reason"] == "token_bound_exceeded"
    with pytest.raises(BudgetHalted):
        await wrap(model, ledger).request(MESSAGES, None, PARAMETERS)
    assert model.calls_count == 1


async def test_unicode_and_output_schema_can_block_request_before_send(tmp_path: Path) -> None:
    messages = [ModelRequest(parts=[UserPromptPart("世界" * 50)])]
    small = ModelRequestParameters()
    large = ModelRequestParameters(
        output_tools=[
            ToolDefinition(
                name="Answers", parameters_json_schema={"type": "object", "description": "schema-content" * 100}
            )
        ]
    )
    small_bound = input_token_upper_bound(messages, {"max_tokens": 100, "max_retries": 0}, small)
    allowance = (small_bound * 1.25 + 100 * 2) / 1_000_000 + 0.00000001
    ledger = BudgetLedger(allowance, tmp_path / "ledger.json")
    provider = StubModel()
    model = wrap(provider, ledger)
    with pytest.raises(BudgetExceeded):
        await model.request(messages, None, large)
    assert provider.calls_count == 0
    await model.request(messages, None, small)
    assert provider.calls_count == 1
    assert not (await ledger.snapshot())["halted"]


@pytest.mark.parametrize("failure", ["transport", "usage_missing", "interrupted_request"])
async def test_explicit_uncertain_resume_preserves_spend_and_persists_audit(tmp_path: Path, failure: str) -> None:
    path = tmp_path / "ledger.json"
    ledger = BudgetLedger(3, path)
    if failure == "interrupted_request":
        await ledger.reserve(label="interrupted", amount=Decimal("0.2"), input_bound=100, output_bound=20)
        ledger = BudgetLedger(3, path)
    elif failure == "transport":
        with pytest.raises(RuntimeError):
            await wrap(StubModel(error=RuntimeError("private-error")), ledger).request(MESSAGES, None, PARAMETERS)
    else:
        await wrap(StubModel(response=ModelResponse(parts=[])), ledger).request(MESSAGES, None, PARAMETERS)
    before = await ledger.snapshot()
    after = await ledger.resume_after_uncertain("request_configuration_corrected")
    assert not after["halted"]
    assert after["charged_usd"] == before["charged_usd"]
    assert after["remaining_usd"] == before["remaining_usd"]
    assert after["limit_usd"] == 3
    assert after["measured_usd"] is None
    assert after["records"] == before["records"]
    assert after["resumptions"][0]["reason_code"] == "request_configuration_corrected"
    reopened = BudgetLedger(3, path)
    await wrap(StubModel(), reopened).request(MESSAGES, None, PARAMETERS)
    final = await reopened.snapshot()
    assert final["charged_usd"] > before["charged_usd"]
    assert final["measured_usd"] is None
    assert final["resumptions"] == after["resumptions"]


async def test_explicit_resume_does_not_restore_spent_allowance(tmp_path: Path) -> None:
    bound = input_token_upper_bound(MESSAGES, {"max_tokens": 100, "max_retries": 0}, PARAMETERS)
    reservation = (bound * 1.25 + 100 * 2) / 1_000_000
    ledger = BudgetLedger(reservation * 1.5, tmp_path / "ledger.json")
    with pytest.raises(RuntimeError):
        await wrap(StubModel(error=RuntimeError("private-error")), ledger).request(MESSAGES, None, PARAMETERS)
    await ledger.resume_after_uncertain("request_configuration_corrected")
    next_model = StubModel()
    with pytest.raises(BudgetExceeded):
        await wrap(next_model, ledger).request(MESSAGES, None, PARAMETERS)
    assert next_model.calls_count == 0
    state = await ledger.snapshot()
    assert state["charged_usd"] >= reservation
    assert state["remaining_usd"] < reservation


@pytest.mark.parametrize("violation", ["token_bound", "reservation"])
async def test_resume_refuses_bound_or_cost_underestimation(tmp_path: Path, violation: str) -> None:
    ledger = BudgetLedger(3, tmp_path / "ledger.json")
    if violation == "token_bound":
        model = StubModel(response=ModelResponse(parts=[], usage=RequestUsage(input_tokens=100, output_tokens=101)))
        await wrap(model, ledger).request(MESSAGES, None, PARAMETERS)
    else:
        request_id = await ledger.reserve(
            label="underestimated", amount=Decimal("0.000000001"), input_bound=1000, output_bound=1000
        )
        usage = usage_record(StubModel().response, stage="test")
        await ledger.settle(request_id, usage=usage, rates=RATES, reservation_rates=RATES)
    before = await ledger.snapshot()
    with pytest.raises(BudgetHalted, match="resumption_not_allowed"):
        await ledger.resume_after_uncertain("request_configuration_corrected")
    assert await ledger.snapshot() == before


async def test_resume_refuses_while_another_request_is_pending(tmp_path: Path) -> None:
    ledger = BudgetLedger(3, tmp_path / "ledger.json")
    pending = StubModel(block=True)
    running = asyncio.create_task(wrap(pending, ledger).request(MESSAGES, None, PARAMETERS))
    await pending.entered.wait()
    try:
        with pytest.raises(RuntimeError):
            await wrap(StubModel(error=RuntimeError("private-error")), ledger).request(MESSAGES, None, PARAMETERS)
        with pytest.raises(BudgetHalted, match="resumption_not_allowed"):
            await ledger.resume_after_uncertain("request_configuration_corrected")
    finally:
        pending.release.set()
        await running
    after = await ledger.resume_after_uncertain("request_configuration_corrected")
    assert not after["halted"]
    assert after["charged_usd"] > 0


async def test_output_bound_quarantine_preserves_cost_and_blocks_only_unsafe_model(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = BudgetLedger(3, path)
    unsafe = StubModel(response=ModelResponse(parts=[], usage=RequestUsage(input_tokens=100, output_tokens=101)))
    await wrap(unsafe, ledger, label="unsafe").request(MESSAGES, None, PARAMETERS)
    before = await ledger.snapshot()
    after = await ledger.quarantine_output_bound_model("unsafe")
    assert after["charged_usd"] == before["charged_usd"]
    assert after["records"] == before["records"]
    assert after["quarantined_labels"] == ["unsafe"]
    reopened = BudgetLedger(3, path)
    blocked = StubModel()
    with pytest.raises(BudgetHalted, match="model_quarantined"):
        await wrap(blocked, reopened, label="unsafe").request(MESSAGES, None, PARAMETERS)
    assert blocked.calls_count == 0
    safe = StubModel()
    await wrap(safe, reopened, label="safe").request(MESSAGES, None, PARAMETERS)
    assert safe.calls_count == 1
    assert (await reopened.snapshot())["charged_usd"] > before["charged_usd"]


@pytest.mark.parametrize("violation", ["wrong_model", "input", "money"])
async def test_quarantine_never_relaxes_other_budget_bounds(tmp_path: Path, violation: str) -> None:
    ledger = BudgetLedger(3, tmp_path / "ledger.json")
    request_id = await ledger.reserve(
        label="unsafe",
        amount=Decimal("0.000000001") if violation == "money" else Decimal("0.01"),
        input_bound=100,
        output_bound=100,
    )
    response = ModelResponse(
        parts=[],
        usage=RequestUsage(
            input_tokens=101 if violation == "input" else 100,
            output_tokens=101,
        ),
    )
    await ledger.settle(request_id, usage=usage_record(response, stage="test"), rates=RATES, reservation_rates=RATES)
    before = await ledger.snapshot()
    with pytest.raises(BudgetHalted, match="quarantine_not_allowed"):
        await ledger.quarantine_output_bound_model("different" if violation == "wrong_model" else "unsafe")
    assert await ledger.snapshot() == before


async def test_visible_only_output_cap_is_refused_without_billable_bound(tmp_path: Path) -> None:
    ledger = BudgetLedger(3, tmp_path / "ledger.json")
    provider = StubModel()
    model = wrap(provider, ledger, total_output_bounded=False)
    with pytest.raises(BudgetHalted, match="total_output_unbounded"):
        await model.request(MESSAGES, None, PARAMETERS)
    assert provider.calls_count == 0
    assert (await ledger.snapshot())["charged_usd"] == 0


async def test_absolute_billable_bound_is_reserved_instead_of_visible_cap(tmp_path: Path) -> None:
    bound = input_token_upper_bound(MESSAGES, {"max_tokens": 100, "max_retries": 0}, PARAMETERS)
    visible_cost = (bound * 1.25 + 100 * 2) / 1_000_000
    absolute_cost = (bound * 1.25 + 1000 * 2) / 1_000_000
    small = BudgetLedger((visible_cost + absolute_cost) / 2, tmp_path / "small.json")
    provider = StubModel()
    with pytest.raises(BudgetExceeded):
        await wrap(provider, small, billable_output_bound=1000).request(MESSAGES, None, PARAMETERS)
    assert provider.calls_count == 0
    ledger = BudgetLedger(3, tmp_path / "large.json")
    response = ModelResponse(parts=[], usage=RequestUsage(input_tokens=100, output_tokens=101))
    await wrap(StubModel(response=response), ledger, billable_output_bound=1000, total_output_bounded=False).request(
        MESSAGES, None, PARAMETERS
    )
    state = await ledger.snapshot()
    assert not state["halted"]
    assert state["records"][0]["max_output_tokens"] == 1000
    assert state["records"][0]["requested_output_tokens"] == 100


def test_blocked_attempts_do_not_dilute_actual_quality_or_become_free_runs(tmp_path: Path) -> None:
    from benchmarks.live_smoke import latest_runs
    from benchmarks.reporting import bootstrap_ci, summarize_rows

    common = {"model": "synthetic", "method": "one_shot", "n": 5}
    events = [
        {**common, "contract_id": "a", "provider_usage": [], "metrics": {"word_f1": 0}, "cost_usd": None},
        {**common, "contract_id": "b", "provider_usage": [], "metrics": {"word_f1": 0}, "cost_usd": None},
        {
            **common,
            "contract_id": "b",
            "provider_usage": [{"input_tokens": 100}],
            "metrics": {"word_f1": 0.8},
            "cost_usd": 0.01,
        },
    ]
    path = tmp_path / "runs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    rows = latest_runs(path)
    summary = summarize_rows(rows, bootstrap_samples=20)[0]
    assert summary["word_f1"] == 0.8
    assert bootstrap_ci(rows, samples=20)["contracts"] == 1
    assert summary["cost_usd"] is None
    assert summary["known_cost_usd"] == 0.01
    assert summary["cost_usd_unknown_records"] == 1


async def test_unknown_bill_reconciliation_only_raises_spend_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = BudgetLedger(3, path)
    with pytest.raises(RuntimeError):
        await wrap(StubModel(error=RuntimeError("private-error")), ledger, label="model").request(
            MESSAGES, None, PARAMETERS
        )
    before = await ledger.snapshot()
    after = await ledger.raise_uncertain_output_bound("model", 1000, RATES)
    assert after["charged_usd"] > before["charged_usd"]
    assert after["remaining_usd"] < before["remaining_usd"]
    assert after["measured_usd"] is None
    assert after["records"][0]["initial_reservation_usd"] == before["records"][0]["reservation_usd"]
    reopened = BudgetLedger(3, path)
    same = await reopened.raise_uncertain_output_bound("model", 100, RATES)
    assert same == after
    await reopened.resume_after_uncertain("configuration_corrected")
    assert (await reopened.snapshot())["charged_usd"] == after["charged_usd"]
