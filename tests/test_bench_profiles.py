"""Profile resolution and SDK wire compatibility; all HTTP goes to MockTransport."""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from typing import Any

import httpx
import openai
import pytest
from benchmarks.live_smoke import MODEL_SPECS, _model, method_config, output_mode_metadata
from benchmarks.methods import _settings
from benchmarks.profiles import model_profile, output_mode_comparison, output_mode_decision, profile_output_mode
from benchmarks.schema import Answers
from pydantic_ai import Agent, PromptedOutput, ToolOutput, models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles.openai import OpenAIModelProfile


@pytest.mark.parametrize(
    "name,expected",
    [
        ("mimo-v2.6-pro", "prompted"),
        ("deepseek-flash", "prompted"),
        ("deepseek-v4-pro", "prompted"),
        ("gpt-6.1-sol", "tools"),
        ("gpt-6-luna", "tools"),
        ("grok-4.7", "tools"),
    ],
)
def test_profile_mode_accounts_for_default_thinking(name: str, expected: str) -> None:
    assert profile_output_mode(model_profile(name)) == expected


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({"thinking": True}, "prompted"),
        ({"thinking": "low"}, "prompted"),
        ({"thinking": False}, "tools"),
        ({"thinking": False, "openai_reasoning_effort": "high"}, "prompted"),
        ({"thinking": True, "openai_reasoning_effort": "none"}, "tools"),
    ],
)
def test_deepseek_request_thinking_overrides_default(settings: dict[str, Any], expected: str) -> None:
    assert profile_output_mode(model_profile("deepseek-flash"), settings) == expected


def test_unconditional_and_thinking_disabling_forcing_restrictions() -> None:
    assert profile_output_mode(model_profile("mimo-v2.6-pro"), {"thinking": False}) == "prompted"
    profile = OpenAIModelProfile(thinking_always_enabled=True, forced_tool_choice_disables_thinking=True)
    assert profile_output_mode(profile, {"thinking": False}) == "prompted"
    assert output_mode_decision(profile, {"thinking": False})["thinking_active"] is True


@pytest.mark.parametrize("name", list(MODEL_SPECS))
@pytest.mark.parametrize("method", ["forkreduce", "one_shot", "tool_agent"])
def test_measured_modes_and_caps_remain_explicitly_preserved(name: str, method: str) -> None:
    config = method_config(name, method)
    expected = "prompted" if (name, method) == ("mimo-v2.6-pro", "forkreduce") else "tools"
    assert config.output_mode == expected
    raised = (name == "mimo-v2.6-pro" and method != "tool_agent") or (
        name == "deepseek-flash" and method == "forkreduce"
    )
    assert config.max_output_tokens == (8192 if raised else 2048)
    decision = output_mode_metadata(name, method, config)
    assert decision["output_mode"] == expected
    if name in {"mimo-v2.6-pro", "deepseek-flash"}:
        assert decision["forced_tool_choice_available"] is False
        assert decision["profile_output_mode"] == "prompted"
        assert decision["source"] == (
            "tool_agent" if method == "tool_agent" else "profile" if expected == "prompted" else "method_configs"
        )


@pytest.mark.parametrize("name", list(MODEL_SPECS))
@pytest.mark.parametrize("method", ["forkreduce", "one_shot", "tool_agent"])
def test_actual_model_and_build_settings_mode_comparison(
    monkeypatch: pytest.MonkeyPatch, name: str, method: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
    config = method_config(name, method)
    model = _model(name, None, max_output_tokens=config.max_output_tokens)
    decision = output_mode_comparison(model, _settings(config))
    expected_profile = "prompted" if name in {"mimo-v2.6-pro", "deepseek-flash"} else "tools"
    expected_explicit = "prompted" if (name, method) == ("mimo-v2.6-pro", "forkreduce") else "tools"
    assert decision["profile_output_mode"] == expected_profile
    assert decision["framework_output_mode"] == expected_profile
    assert decision["output_modes_equivalent"] is True
    assert decision["framework_error_code"] is None
    # Diagnostics must not replace historical tools overrides with profile-derived prompted.
    assert config.output_mode == expected_explicit


@pytest.mark.parametrize(
    "settings,benchmark_mode,framework_mode,error_code",
    [
        ({"thinking": True}, "prompted", "prompted", None),
        ({"thinking": False}, "tools", "tools", None),
        ({"thinking": True, "openai_reasoning_effort": "none"}, "tools", "tools", None),
        ({"thinking": False, "openai_reasoning_effort": "high"}, "prompted", "prompted", None),
        ({"anthropic_thinking": {"type": "enabled", "budget_tokens": 1024}}, "tools", "prompted", None),
        ({"google_thinking_config": {"thinking_budget": 1024}}, "tools", "prompted", None),
        ({"google_thinking_config": {"thinking_level": "minimal"}}, "tools", "prompted", None),
        ({"extra_body": {"thinking": {"type": "enabled"}}}, "tools", "prompted", None),
        ({"extra_body": {"reasoning_effort": "high"}}, "tools", "prompted", None),
        ({"extra_body": {"enable_thinking": True}}, "tools", "prompted", None),
        ({"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}, "tools", "prompted", None),
        ({"chat_template_kwargs": {"enable_thinking": True}}, "tools", "prompted", None),
        (
            {"openai_reasoning_effort": "none", "anthropic_thinking": {"type": "adaptive"}},
            "tools",
            "prompted",
            None,
        ),
        ({"openai_reasoning_effort": "future"}, "prompted", None, "unknown_thinking_control"),
        ({"extra_body": {"reasoning_switch": True}}, "tools", None, "unknown_thinking_control"),
    ],
)
def test_framework_supported_setting_differences_remain_diagnostic(
    settings: dict[str, Any], benchmark_mode: str, framework_mode: str | None, error_code: str | None
) -> None:
    profile = OpenAIModelProfile(supports_thinking=True, supports_forced_tool_choice_with_thinking=False)
    model = TestModel(profile=profile)
    decision = output_mode_comparison(model, settings)
    assert decision["profile_output_mode"] == benchmark_mode
    assert decision["framework_output_mode"] == framework_mode
    assert decision["output_modes_equivalent"] is (benchmark_mode == framework_mode)
    assert decision["framework_error_code"] == error_code


@pytest.mark.parametrize(
    "profile,settings,benchmark_mode,framework_mode",
    [
        (OpenAIModelProfile(supports_tools=False), {}, "tools", "prompted"),
        (
            OpenAIModelProfile(supports_thinking=False, supports_forced_tool_choice_with_thinking=False),
            {"thinking": True},
            "prompted",
            "tools",
        ),
        (
            OpenAIModelProfile(thinking_always_enabled=True, supports_forced_tool_choice_with_thinking=False),
            {"openai_reasoning_effort": "none"},
            "tools",
            "prompted",
        ),
        (
            OpenAIModelProfile(
                supports_thinking=True,
                thinking_enabled_by_default=True,
                supports_forced_tool_choice_with_thinking=False,
            ),
            {"extra_body": {"thinking": {"type": "disabled"}}},
            "prompted",
            "tools",
        ),
    ],
)
def test_profile_restrictions_and_provider_disable_controls_are_not_equivalent(
    profile: OpenAIModelProfile, settings: dict[str, Any], benchmark_mode: str, framework_mode: str
) -> None:
    decision = output_mode_comparison(TestModel(profile=profile), settings)
    assert decision["profile_output_mode"] == benchmark_mode
    assert decision["framework_output_mode"] == framework_mode
    assert decision["output_modes_equivalent"] is False
    assert decision["framework_error_code"] is None


def test_comparison_shallow_merges_model_then_build_settings_without_exposing_them() -> None:
    profile = OpenAIModelProfile(supports_thinking=True, supports_forced_tool_choice_with_thinking=False)
    model_settings = {
        "thinking": True,
        "openai_reasoning_effort": "high",
        "extra_body": {"reasoning_effort": "high"},
        "extra_headers": {"Authorization": "offline-private-placeholder"},
    }
    build_settings = {
        "thinking": False,
        "openai_reasoning_effort": None,
        "extra_body": {"enable_thinking": False},
    }
    model = TestModel(profile=profile, settings=model_settings)
    before = output_mode_comparison(model)
    assert before["profile_output_mode"] == before["framework_output_mode"] == "prompted"
    after = output_mode_comparison(model, build_settings)
    assert after["profile_output_mode"] == after["framework_output_mode"] == "tools"
    assert after["thinking_active"] is False
    assert after["output_modes_equivalent"] is True
    assert model_settings["extra_body"] == {"reasoning_effort": "high"}
    assert build_settings["extra_body"] == {"enable_thinking": False}
    assert "offline-private-placeholder" not in json.dumps(after)
    assert "extra_headers" not in after
    assert "extra_body" not in after


def _legacy_profile(name: str) -> OpenAIModelProfile | None:
    # Pre-profile-migration behavior, not a comparison of framework versions.
    if name != "deepseek-flash":
        return None
    return OpenAIModelProfile(
        supports_thinking=True,
        thinking_enabled_by_default=True,
        supports_forced_tool_choice=False,
        openai_chat_thinking_field="reasoning_content",
        openai_chat_send_back_thinking_parts="field",
        openai_chat_thinking_requires_field_on_tool_calls=True,
        openai_chat_supports_max_completion_tokens=False,
    )


async def _wire(
    monkeypatch: pytest.MonkeyPatch, name: str, method: str, *, legacy: bool, history: str = "none"
) -> bytes:
    captured: list[bytes] = []
    config = method_config(name, method)

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        args = '{"answers":[{"question_id":"probe","spans":[],"source_ids":[]}]}'
        if MODEL_SPECS[name]["api"] == "responses":
            output = [
                {"type": "function_call", "id": "fc_probe", "call_id": "probe", "name": "Answers", "arguments": args}
            ]
            return httpx.Response(
                200,
                json={
                    "id": "resp_probe",
                    "object": "response",
                    "created_at": 0,
                    "model": name,
                    "status": "completed",
                    "output": output,
                    "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
                },
            )
        message: dict[str, Any] = {"role": "assistant", "content": args}
        if config.output_mode == "tools":
            message.update(
                content=None,
                tool_calls=[{"id": "probe", "type": "function", "function": {"name": "Answers", "arguments": args}}],
            )
        return httpx.Response(
            200,
            json={
                "id": "chat_probe",
                "object": "chat.completion",
                "created": 0,
                "model": name,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        )

    original_client = openai.AsyncOpenAI
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        with monkeypatch.context() as patch:
            patch.setenv("OPENAI_API_KEY", "offline-placeholder")
            patch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
            patch.setattr(openai, "AsyncOpenAI", lambda **kwargs: original_client(**kwargs, http_client=http_client))
            if legacy:
                patch.setattr("benchmarks.live_smoke.model_profile", _legacy_profile)
            model = _model(name, None).wrapped
            output_type = (
                ToolOutput(Answers, name="Answers", max_retries=0)
                if config.output_mode == "tools"
                else PromptedOutput(Answers)
            )
            agent = Agent(model, output_type=output_type, retries=0)
            messages = []
            if history != "none":
                parts = [ToolCallPart("lookup", {}, tool_call_id="prior")]
                if history == "thinking":
                    parts.insert(
                        0, ThinkingPart("Public synthetic reasoning.", id="reasoning_content", provider_name="openai")
                    )
                messages = [
                    ModelRequest(parts=[UserPromptPart("Prior public lookup.")]),
                    ModelResponse(parts=parts, model_name=name, provider_name="openai"),
                    ModelRequest(parts=[ToolReturnPart("lookup", "Public result.", tool_call_id="prior")]),
                ]
            expected_warning = (
                "Sampling parameters ['temperature'] are not supported when reasoning is enabled. "
                "These settings will be ignored."
            )
            warning_context = (
                pytest.warns(UserWarning, match=f"^{re.escape(expected_warning)}$")
                if name in {"gpt-6.1-sol", "gpt-6-luna"}
                else nullcontext()
            )
            with models.override_allow_model_requests(True), warning_context as emitted:
                result = await agent.run(
                    "Return probe with no spans.",
                    message_history=messages,
                    model_settings={"max_tokens": config.max_output_tokens, "temperature": config.temperature},
                )
            if emitted is not None:
                assert len(emitted) == 1
                assert str(emitted[0].message) == expected_warning
            assert result.output.answers[0].question_id == "probe"
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("name", list(MODEL_SPECS))
@pytest.mark.parametrize("method", ["forkreduce", "one_shot", "tool_agent"])
async def test_current_config_wire_bytes(monkeypatch: pytest.MonkeyPatch, name: str, method: str) -> None:
    before = await _wire(monkeypatch, name, method, legacy=True)
    after = await _wire(monkeypatch, name, method, legacy=False)
    body = json.loads(after)
    cap_field = (
        "max_output_tokens"
        if MODEL_SPECS[name]["api"] == "responses"
        else ("max_tokens" if name == "deepseek-flash" else "max_completion_tokens")
    )
    assert body[cap_field] == method_config(name, method).max_output_tokens
    if name == "mimo-v2.6-pro" and method != "forkreduce":
        assert json.loads(before)["tool_choice"] == "required"
        assert body["tool_choice"] == "auto"
        assert after == before.replace(b'"tool_choice":"required"', b'"tool_choice":"auto"')
    else:
        assert before == after


@pytest.mark.parametrize("name", ["mimo-v2.6-pro", "deepseek-flash"])
@pytest.mark.parametrize("history", ["thinking", "empty"])
async def test_reasoning_field_send_back(monkeypatch: pytest.MonkeyPatch, name: str, history: str) -> None:
    before = await _wire(monkeypatch, name, "one_shot", legacy=True, history=history)
    after = await _wire(monkeypatch, name, "one_shot", legacy=False, history=history)
    old, new = json.loads(before), json.loads(after)
    assistant = next(message for message in new["messages"] if message["role"] == "assistant")
    assert assistant["reasoning_content"] == ("Public synthetic reasoning." if history == "thinking" else "")
    if name == "deepseek-flash":
        assert before == after
    else:
        old["tool_choice"] = "auto"
        if history == "empty":
            prior = next(message for message in old["messages"] if message["role"] == "assistant")
            assert "reasoning_content" not in prior
            prior["reasoning_content"] = ""
        assert old == new
