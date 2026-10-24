"""Private, opt-in failed-response JSONL diagnostics; never serialize request data."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_ai.messages import TextPart, ToolCallPart

from benchmarks.schema import Answers

MAX_RESPONSE_CHARS = 2000
_SECRET_ENV = re.compile(
    r"(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION|PRIVATE[_-]?KEY|ACCESS[_-]?KEY)", re.I
)
_KEY_LIKE = re.compile(
    r"\b(?:(?:sk|pk|rk)[-_][A-Za-z0-9_-]{6,}|AIza[A-Za-z0-9_-]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|"
    r"gh[pousr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"hf_[A-Za-z0-9]{10,}|glpat-[A-Za-z0-9_-]{10,})\b"
)
_LABELED_SECRET = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|authorization|password|secret|token)"
    r"[\"']?\s*[:=]\s*[\"']?)([^\s\"',;}]+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_REASONS = {
    "validation",
    "question_ids",
    "reduce_provenance",
    "citation_invalid",
    "check_failed",
    "wrong_output_type",
    "missing_output",
    "repair_apply",
    "length",
    "no_tool",
    "malformed_json",
    "schema_validation",
    "provider_error",
    "budget_refusal",
    "budget_exceeded",
    "question_budget",
    "transport",
    "context_overflow",
    "user_error",
    "unexpected",
    "reduce_failed",
    "reduce_did_not_shrink",
}
_CLASSIFICATIONS = {
    "validation",
    "question_ids",
    "reduce_provenance",
    "citation_invalid",
    "length",
    "no_tool",
    "malformed_json",
    "schema_validation",
    "provider_error",
    "budget_refusal",
}


def redact(value: str) -> str:
    """Remove live credential values as well as common credential-shaped strings."""
    secrets = sorted(
        {value for name, value in os.environ.items() if value and _SECRET_ENV.search(name)},
        key=len,
        reverse=True,
    )
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    value = _KEY_LIKE.sub("[REDACTED]", value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    return _LABELED_SECRET.sub(lambda match: match[1] + "[REDACTED]", value)


def safe_reason(reason: str) -> str:
    return reason if reason in _REASONS or re.fullmatch(r"http_[0-9]{3}", reason) else "validation"


def response_summary(response: Any) -> dict[str, Any]:
    """Safe outcome metadata, including for successful responses; no raw payloads."""
    finish = getattr(response, "finish_reason", None)
    return {
        "finish_reason": redact(finish)[:80] if isinstance(finish, str) else None,
        "tool_call_emitted": any(isinstance(part, ToolCallPart) for part in getattr(response, "parts", ())),
        "tool_names": [
            redact(part.tool_name)[:80] for part in getattr(response, "parts", ()) if isinstance(part, ToolCallPart)
        ],
    }


def _payload(part: ToolCallPart) -> str:
    if isinstance(part.args, str):
        return part.args
    return json.dumps(part.args, ensure_ascii=False, separators=(",", ":"))


def _classification(response: Any, *, reason: str, expected_tool: bool | None) -> tuple[str, str, str]:
    parts = response.parts
    tools = [part for part in parts if isinstance(part, ToolCallPart)]
    text = "".join(part.content for part in parts if isinstance(part, TextPart))
    payloads = [_payload(part) for part in tools] if tools else [text]
    json_status, schema_status = "valid_json", "valid"
    for payload in payloads:
        try:
            decoded = json.loads(payload)
        except (ValueError, TypeError):
            json_status, schema_status = "malformed_json", "not_checked"
            break
        try:
            Answers.model_validate(decoded)
        except ValidationError:
            schema_status = "invalid"
    if getattr(response, "finish_reason", None) in {"length", "max_tokens", "max_output_tokens"}:
        classification = "length"
    elif expected_tool and not tools:
        classification = "no_tool"
    elif json_status == "malformed_json":
        classification = "malformed_json"
    elif schema_status == "invalid":
        classification = "schema_validation"
    elif reason in {"question_ids", "reduce_provenance", "citation_invalid"}:
        classification = reason
    else:
        classification = "validation"
    return classification, json_status, schema_status


def failure_record(
    response: Any,
    *,
    stage: str,
    reason: str,
    reasoning_tokens: int | None = None,
    expected_tool: bool | None = None,
    classification: str | None = None,
) -> dict[str, Any]:
    """Build only from visible model output, excluding ThinkingPart and metadata."""
    reason = safe_reason(reason)
    derived, json_status, schema_status = _classification(response, reason=reason, expected_tool=expected_tool)
    record: dict[str, Any] = {
        "stage": stage if stage in {"extraction", "reduce", "citation_repair", "warmup"} else "extraction",
        "failure_reason": reason,
        "classification": classification if classification in _CLASSIFICATIONS - {"validation"} else derived,
        "json_classification": json_status,
        "schema_classification": schema_status,
        "finish_reason": response_summary(response)["finish_reason"],
        "reasoning_tokens": reasoning_tokens,
        "tool_call_emitted": any(isinstance(part, ToolCallPart) for part in response.parts),
        "tools": [],
        "text": "",
        "truncated": False,
    }
    # Allocate a single budget across fields; final JSON encoding is bounded too.
    remaining = MAX_RESPONSE_CHARS
    for part in response.parts:
        if isinstance(part, ToolCallPart):
            if remaining <= 0 or len(record["tools"]) >= 16:
                record["truncated"] = True
                continue
            name, args = redact(part.tool_name), redact(_payload(part))
            name_kept = name[:remaining]
            remaining -= len(name_kept)
            args_kept = args[:remaining]
            remaining -= len(args_kept)
            record["truncated"] |= len(name_kept) < len(name) or len(args_kept) < len(args)
            record["tools"].append({"name": name_kept, "arguments": args_kept})
        elif isinstance(part, TextPart):
            text = redact(part.content)
            kept = text[:remaining]
            remaining -= len(kept)
            record["text"] += kept
            record["truncated"] |= len(kept) < len(text)
    return record


def exception_record(exc: Exception, *, stage: str) -> dict[str, Any]:
    # Local import avoids the budget module's dependency on providers. Never read
    # exception args, message, response body, headers, request, or traceback.
    from benchmarks.budget import BudgetExceeded, BudgetHalted

    budget = isinstance(exc, (BudgetExceeded, BudgetHalted))
    status = getattr(exc, "status_code", None)
    return {
        "stage": stage if stage in {"extraction", "reduce", "citation_repair", "warmup"} else "extraction",
        "failure_reason": "budget_refusal" if budget else "provider_error",
        "classification": "budget_refusal" if budget else "provider_error",
        "exception_type": redact(type(exc).__name__)[:80],
        "status_code": (
            status if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599 else None
        ),
    }


def append_failure(path: str | Path, record: dict[str, Any]) -> None:
    """Bound the entire JSON record, not merely each field or output part."""

    def encoded() -> str:
        return json.dumps(record, ensure_ascii=False, separators=(",", ":"))

    line = encoded()
    while len(line) > MAX_RESPONSE_CHARS:
        record["truncated"] = True
        fields = [(record, "text")]
        fields.extend((tool, key) for tool in record.get("tools", []) for key in ("name", "arguments"))
        owner, key = max(fields, key=lambda entry: len(entry[0].get(entry[1], "")))
        value = owner.get(key, "")
        if value:
            owner[key] = value[: max(0, len(value) - max(1, (len(line) - MAX_RESPONSE_CHARS) // 6))]
        elif record.get("tools"):
            record["tools"].pop()
        else:
            raise ValueError("diagnostic_metadata_too_large")
        line = encoded()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
