"""Fail-closed, request-level spending ledger for the opt-in text smoke benchmark.

Input reservations count UTF-8 bytes of serialized messages, settings and schemas
as tokens, plus a framing allowance. This is deliberately much larger than a
normal tokenizer estimate, but is NOT a provider-guaranteed upper bound: it
assumes text-only byte-based tokenization and bounded provider framing. Likewise,
a reservation price ceiling must really bound the selected model's prices.
Provider-side spending limits remain the only independent hard monetary limit.

One ledger is shared by every model/request in one process. Use one process per
ledger path, located in ignored ``benchmarks/results/`` by the CLI. Existing
ledgers resume their allowance; an interrupted request is charged its entire
reservation and halts further calls. A caller may explicitly acknowledge an
investigated uncertainty after its full reservation was charged; this is audited
and never refunds its charge. No prompts, responses, headers, credentials, or raw
provider exceptions are persisted.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_core import to_json

from benchmarks.estimate import price_usage
from benchmarks.metrics import summarize_usage
from benchmarks.providers import usage_record

INPUT_BOUND_CAVEAT = (
    "UTF-8 serialized text bytes plus framing/schema allowance is a conservative estimate, "
    "not a provider-guaranteed tokenizer or dollar upper bound; price ceilings must bound actual rates."
)
REQUEST_OVERHEAD_TOKENS = 8192
MESSAGE_OVERHEAD_TOKENS = 256
_MONEY_UNIT = Decimal("0.000000001")
_RATE_KEYS = ("input_miss", "input_hit", "cache_write", "output")


class BudgetExceeded(RuntimeError):
    """A request was refused before sending because its reservation would overflow."""


class BudgetHalted(RuntimeError):
    """Uncertain billing or an underestimated reservation prevents further sends."""


def _money(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("invalid_budget_amount")
    try:
        amount = Decimal(str(value))
    except Exception as exc:
        raise ValueError("invalid_budget_amount") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("invalid_budget_amount")
    return amount


def _rounded(value: Decimal) -> Decimal:
    return value.quantize(_MONEY_UNIT, rounding=ROUND_CEILING)


def _rates(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: float(_money(value[key])) for key in _RATE_KEYS}
    if any(not math.isfinite(rate) for rate in result.values()):
        raise ValueError("invalid_budget_rate")
    result["cache_policy"] = value.get("cache_policy", "automatic")
    return result


def _uncached_cost(input_tokens: int, output_tokens: int, rates: Mapping[str, Any]) -> Decimal:
    # Explicit cache writes can cost more than uncached input. Never assume a hit
    # or a discount, even if a provider happens to create/read a cache entry.
    input_rate = max(_money(rates[key]) for key in ("input_miss", "input_hit", "cache_write"))
    return _rounded((input_tokens * input_rate + output_tokens * _money(rates["output"])) / 1_000_000)


def input_token_upper_bound(messages: Any, settings: Mapping[str, Any], parameters: Any) -> int:
    """Count serialized UTF-8 bytes, including tool/output schemas, without logging them.

    Only text requests without provider-native paid tools are supported. Transport
    options are not token input. Unknown serialization fails before any request.
    See ``INPUT_BOUND_CAVEAT`` for the byte/framing assumptions.
    """
    if getattr(parameters, "native_tools", ()) or getattr(parameters, "allow_image_output", False):
        raise ValueError("budget_text_requests_only")
    for message in messages:
        for part in message.parts:
            if (
                isinstance(part, UserPromptPart)
                and not isinstance(part.content, str)
                and (
                    not isinstance(part.content, (list, tuple))
                    or any(not isinstance(item, str) for item in part.content)
                )
            ):
                raise ValueError("budget_text_requests_only")
    body_settings = {
        key: value
        for key, value in settings.items()
        if key not in {"extra_headers", "timeout", "request_timeout", "max_retries"}
    }
    serialized = to_json({"messages": messages, "settings": body_settings, "parameters": parameters})
    parts = sum(len(message.parts) for message in messages)
    return len(serialized) + REQUEST_OVERHEAD_TOKENS + MESSAGE_OVERHEAD_TOKENS * (len(messages) + parts)


class BudgetLedger:
    """An atomic asyncio ledger shared across models and resumed across invocations.

    ``charged_usd`` includes observed official cost or a conservative charge when
    cost is unknown. ``measured_usd`` is None if any completed request has unknown
    official cost, and must not be presented as measured zero in that case.
    """

    def __init__(self, limit_usd: float, ledger_path: Path) -> None:
        self.limit = _money(limit_usd)
        self.path = Path(ledger_path)
        self._lock = asyncio.Lock()
        self._charged = Decimal(0)
        self._measured = Decimal(0)
        self._measurement_unknown = False
        self._pending: dict[int, Decimal] = {}
        self._records: list[dict[str, Any]] = []
        self._resumptions: list[dict[str, Any]] = []
        self._quarantined_labels: set[str] = set()
        self._halt_reason: str | None = None
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("version") != 1:
                raise ValueError("budget_ledger_version_mismatch")
            if _money(saved["limit_usd"]) != self.limit:
                raise ValueError("budget_limit_mismatch")
            self._charged = _money(saved["charged_usd"])
            self._measured = _money(saved["measured_known_usd"])
            self._measurement_unknown = bool(saved["measurement_unknown"])
            self._records = saved["records"]
            self._resumptions = saved.get("resumptions", [])
            self._quarantined_labels = set(saved.get("quarantined_labels", []))
            self._halt_reason = saved["halt_reason"]
            for record in self._records:
                if record["status"] == "reserved":
                    charge = _money(record["reservation_usd"])
                    self._charged += charge
                    record.update(
                        status="uncertain", charged_usd=float(charge), reason="interrupted_request", measured_usd=None
                    )
                    self._measurement_unknown = True
                    self._halt_reason = "interrupted_request"
            if self._charged > self.limit:
                self._halt_reason = "budget_limit_exceeded"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._persist()

    def _snapshot(self) -> dict[str, Any]:
        reserved = sum(self._pending.values(), Decimal(0))
        return {
            "version": 1,
            "limit_usd": float(self.limit),
            "charged_usd": float(self._charged),
            "reserved_usd": float(reserved),
            "remaining_usd": float(max(Decimal(0), self.limit - self._charged - reserved)),
            "measured_usd": None if self._measurement_unknown else float(self._measured),
            "measured_known_usd": float(self._measured),
            "measurement_unknown": self._measurement_unknown,
            "halted": self._halt_reason is not None,
            "halt_reason": self._halt_reason,
            "input_bound_caveat": INPUT_BOUND_CAVEAT,
            "records": copy.deepcopy(self._records),
            "resumptions": copy.deepcopy(self._resumptions),
            "quarantined_labels": sorted(self._quarantined_labels),
        }

    def _persist(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self._snapshot(), handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            self._halt_reason = "ledger_persistence_failed"
            raise

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return self._snapshot()

    async def raise_uncertain_output_bound(
        self, label: str, billable_output_bound: int, rates: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Monotonically reconcile unknown bills after discovering a larger cap.

        This is not a new request or a refund. Original reservations are retained
        in the audit, measured cost stays unknown, and an exhausted cap halts.
        """
        if (
            not isinstance(billable_output_bound, int)
            or isinstance(billable_output_bound, bool)
            or billable_output_bound < 1
        ):
            raise ValueError("invalid_billable_output_bound")
        ceiling = _rates(rates)
        async with self._lock:
            if self._pending:
                raise BudgetHalted("budget_reconciliation_while_pending")
            delta = Decimal(0)
            changed = []
            for record in self._records:
                if record["label"] != label or record["status"] != "uncertain":
                    continue
                bound = max(record["max_output_tokens"], billable_output_bound)
                new_charge = max(
                    _money(record["charged_usd"]),
                    _uncached_cost(record["input_token_bound"], bound, ceiling),
                )
                increase = new_charge - _money(record["charged_usd"])
                if increase:
                    record.setdefault("initial_reservation_usd", record["reservation_usd"])
                    record.setdefault("initial_max_output_tokens", record["max_output_tokens"])
                    record.update(
                        max_output_tokens=bound, reservation_usd=float(new_charge), charged_usd=float(new_charge)
                    )
                    delta += increase
                    changed.append(record["request_id"])
            if changed:
                self._charged += delta
                self._resumptions.append(
                    {
                        "action": "raise_uncertain_output_bound",
                        "label": label,
                        "request_ids": changed,
                        "additional_charge_usd": float(delta),
                        "reason_code": "official_absolute_output_ceiling",
                    }
                )
                if self._charged > self.limit:
                    self._halt_reason = "budget_limit_exceeded"
                self._persist()
            return self._snapshot()

    async def resume_after_uncertain(self, reason_code: str) -> dict[str, Any]:
        """Explicitly resume after investigating a fully charged uncertain request.

        Every uncertain request retains its full conservative reservation charge.
        This never retries a request, changes the limit, clears unknown measured
        cost, or resumes token-bound, underestimated-cost or persistence failures.
        ``reason_code`` must describe the correction as a public-safe machine code.
        """
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", reason_code):
            raise ValueError("invalid_budget_resumption_reason")
        async with self._lock:
            allowed = {"request_without_response", "usage_missing", "interrupted_request"}
            if self._halt_reason not in allowed or self._pending:
                raise BudgetHalted("budget_resumption_not_allowed")
            uncertain = [record for record in self._records if record["status"] == "uncertain"]
            if not uncertain or any(
                _money(record["charged_usd"]) < _money(record["reservation_usd"]) for record in uncertain
            ):
                raise BudgetHalted("budget_uncertain_reservation_not_fully_charged")
            self._resumptions.append(
                {
                    "reason_code": reason_code,
                    "action": "explicit_resume_after_fully_charged_uncertainty",
                    "previous_halt_reason": self._halt_reason,
                    "after_request_id": uncertain[-1]["request_id"],
                    "charged_usd": float(self._charged),
                }
            )
            self._halt_reason = None
            self._persist()
            return self._snapshot()

    async def quarantine_output_bound_model(self, label: str) -> dict[str, Any]:
        """Retire an unsafe model without disabling independently bounded models.

        Only observed output-bound violations whose known cost fit the original
        reservation qualify. Input/tokenizer and monetary-bound violations still
        halt globally. The unsafe label can never send another request here.
        """
        async with self._lock:
            violations = [
                record
                for record in self._records
                if record.get("usage")
                and not record["usage"].get("usage_missing", True)
                and (
                    record["usage"]["output_tokens"] > record["max_output_tokens"]
                    or record["usage"]["input_tokens"] > record["input_token_bound"]
                )
            ]
            if (
                self._halt_reason != "token_bound_exceeded"
                or self._pending
                or self._charged >= self.limit
                or not violations
                or any(
                    record["label"] != label
                    or record["usage"]["input_tokens"] > record["input_token_bound"]
                    or record["measured_usd"] is None
                    or _money(record["charged_usd"]) > _money(record["reservation_usd"])
                    for record in violations
                )
            ):
                raise BudgetHalted("budget_model_quarantine_not_allowed")
            self._quarantined_labels.add(label)
            self._resumptions.append(
                {
                    "action": "quarantine_output_bound_model",
                    "previous_halt_reason": self._halt_reason,
                    "label": label,
                    "after_request_id": violations[-1]["request_id"],
                    "charged_usd": float(self._charged),
                }
            )
            self._halt_reason = None
            self._persist()
            return self._snapshot()

    async def reserve(
        self,
        *,
        label: str,
        amount: Decimal,
        input_bound: int,
        output_bound: int,
        requested_output_tokens: int | None = None,
    ) -> int:
        async with self._lock:
            if self._halt_reason is not None:
                raise BudgetHalted("budget_halted")
            if label in self._quarantined_labels:
                raise BudgetHalted("budget_model_quarantined")
            reserved = sum(self._pending.values(), Decimal(0))
            request_id = len(self._records)
            record = {
                "request_id": request_id,
                "label": label,
                "reservation_usd": float(amount),
                "input_token_bound": input_bound,
                "max_output_tokens": output_bound,
                "requested_output_tokens": requested_output_tokens
                if requested_output_tokens is not None
                else output_bound,
                "charged_usd": 0.0,
                "measured_usd": None,
                "status": "reserved",
            }
            self._records.append(record)
            if self._charged + reserved + amount > self.limit:
                record.update(status="refused", reason="reservation_exceeds_remaining")
                self._persist()
                raise BudgetExceeded("budget_reservation_exceeds_remaining")
            self._pending[request_id] = amount
            self._persist()
            return request_id

    async def settle(
        self,
        request_id: int,
        *,
        usage: Mapping[str, Any] | None,
        rates: Mapping[str, Any] | None,
        reservation_rates: Mapping[str, Any],
    ) -> None:
        async with self._lock:
            reservation = self._pending.pop(request_id)
            record = self._records[request_id]
            measured: float | None = None
            if usage is None or usage["usage_missing"]:
                charge = reservation
                reason = "request_without_response" if usage is None else "usage_missing"
                self._halt_reason = reason
                status = "uncertain"
            else:
                summary = summarize_usage([usage])
                if rates is not None:
                    measured = price_usage(summary, rates)
                if measured is not None:
                    charge = _rounded(_money(measured))
                    reason = "official_usage"
                else:
                    charge = _uncached_cost(usage["input_tokens"], usage["output_tokens"], reservation_rates)
                    reason = "conservative_usage_unknown_official_cost"
                status = "settled"
                if charge > reservation:
                    self._halt_reason = "reservation_underestimated"
                # Token-bound violations also halt, even if a price discount
                # happens to keep the current request cheaper than reserved.
                if (
                    usage["input_tokens"] > record["input_token_bound"]
                    or usage["output_tokens"] > record["max_output_tokens"]
                ):
                    self._halt_reason = "token_bound_exceeded"
            self._charged += charge
            if measured is None:
                self._measurement_unknown = True
            else:
                self._measured += _money(measured)
            record.update(status=status, charged_usd=float(charge), measured_usd=measured, reason=reason)
            if usage is not None:
                record["usage"] = dict(usage)
            self._persist()


class BudgetModel(WrapperModel):
    """Reserve each nonstreaming model request before invoking the provider.

    The caller must verify that its provider caps *total billable* output, not
    merely visible text. Otherwise set ``total_output_bounded=False`` to refuse
    sends, or supply an official absolute ``billable_output_bound`` for reservation.
    Unknown prices require an explicit conservative ``reservation_rates`` ceiling.
    """

    def __init__(
        self,
        wrapped: Model,
        ledger: BudgetLedger,
        *,
        rates: Mapping[str, Any] | None,
        reservation_rates: Mapping[str, Any] | None = None,
        max_output_tokens: int,
        label: str,
        billable_output_bound: int | None = None,
        total_output_bounded: bool = True,
    ) -> None:
        super().__init__(wrapped)
        if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
            raise ValueError("invalid_budget_output_bound")
        if billable_output_bound is not None and (
            not isinstance(billable_output_bound, int)
            or isinstance(billable_output_bound, bool)
            or billable_output_bound < max_output_tokens
        ):
            raise ValueError("invalid_billable_output_bound")
        if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", label):
            raise ValueError("invalid_budget_label")
        if rates is None and reservation_rates is None:
            raise ValueError("budget_reservation_rates_required")
        self.ledger = ledger
        self.rates = _rates(rates) if rates is not None else None
        self.reservation_rates = _rates(reservation_rates) if reservation_rates is not None else self.rates
        assert self.reservation_rates is not None
        if self.rates is not None and any(self.reservation_rates[key] < self.rates[key] for key in _RATE_KEYS):
            raise ValueError("budget_reservation_rates_below_official")
        self.max_output_tokens = max_output_tokens
        self.budget_label = label
        self.billable_output_bound = billable_output_bound
        self.total_output_bounded = total_output_bounded

    async def request(self, messages: Any, model_settings: Any, model_request_parameters: Any) -> Any:
        if not self.total_output_bounded and self.billable_output_bound is None:
            raise BudgetHalted("budget_total_output_unbounded")
        settings = {**(self.wrapped.settings or {}), **(model_settings or {})}
        requested_output = settings.get("max_tokens", self.max_output_tokens)
        if not isinstance(requested_output, int) or isinstance(requested_output, bool) or requested_output <= 0:
            raise ValueError("invalid_budget_output_bound")
        output_bound = min(requested_output, self.max_output_tokens)
        settings["max_tokens"] = output_bound
        settings["max_retries"] = 0
        if settings.get("openai_background"):
            raise ValueError("budget_background_requests_unsupported")
        if settings.get("extra_body"):
            extra = dict(settings["extra_body"])
            if extra.get("n", 1) != 1:
                raise ValueError("budget_multiple_completions_unsupported")
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                if key in extra:
                    extra[key] = output_bound
            settings["extra_body"] = extra
        input_bound = input_token_upper_bound(messages, settings, model_request_parameters)
        billable_bound = self.billable_output_bound or output_bound
        reservation = _uncached_cost(input_bound, billable_bound, self.reservation_rates)
        request_id = await self.ledger.reserve(
            label=self.budget_label,
            amount=reservation,
            input_bound=input_bound,
            output_bound=billable_bound,
            requested_output_tokens=output_bound,
        )
        try:
            response = await self.wrapped.request(messages, settings, model_request_parameters)
            record = usage_record(response, stage="request")
        except BaseException:
            await asyncio.shield(
                self.ledger.settle(request_id, usage=None, rates=self.rates, reservation_rates=self.reservation_rates)
            )
            raise
        await asyncio.shield(
            self.ledger.settle(request_id, usage=record, rates=self.rates, reservation_rates=self.reservation_rates)
        )
        return response

    async def count_tokens(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("budget_provider_token_count_unsupported")

    async def compact_messages(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("budget_provider_compaction_unsupported")

    @asynccontextmanager
    async def request_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise ValueError("budget_streaming_unsupported")
        yield  # pragma: no cover -- makes this an async context manager, without sending a request
