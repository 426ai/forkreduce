"""Offline output capability, frozen-mode, diagnostics, and result contracts."""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

from forkreduce import (
    Budget,
    ForkReduceError,
    OutputModeConflictError,
    Policy,
    Query,
    UnforcedToolsError,
    UnforcedToolsWarning,
    build,
)
from forkreduce.documents import coerce_documents


class Answer(BaseModel):
    """A structured answer."""

    model_config = ConfigDict(extra="forbid")
    value: int


class Note(BaseModel):
    """A note."""

    text: str


def session_for(model: Any, **kwargs: Any) -> Any:
    return build(
        model,
        documents="α shared evidence",
        instructions="Return grounded answers.",
        outputs=[Answer, Note],
        warmup=False,
        policy=kwargs.pop("policy", Policy(retries=0, retry_backoff=0)),
        **kwargs,
    )


def recording_model(
    *, profile: dict[str, Any] | None = None, settings: dict[str, Any] | None = None
) -> tuple[FunctionModel, list[AgentInfo]]:
    calls: list[AgentInfo] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(info)
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart("Answer", {"value": 7})])
        return ModelResponse(parts=[TextPart('{"value":7}')])

    return (
        FunctionModel(respond, profile=cast(ModelProfile, profile or {}), settings=cast(ModelSettings, settings)),
        calls,
    )


@pytest.mark.parametrize(
    ("profile", "mode"),
    [
        ({}, "tools"),
        ({"supports_tools": False}, "prompted"),
        ({"supports_forced_tool_choice": False}, "prompted"),
        ({"default_structured_output_mode": "native"}, "tools"),
        ({"default_structured_output_mode": "prompted"}, "tools"),
        ({"thinking_enabled_by_default": True, "supports_forced_tool_choice_with_thinking": False}, "prompted"),
        ({"thinking_enabled_by_default": True, "forced_tool_choice_disables_thinking": True}, "prompted"),
        ({"thinking_enabled_by_default": True}, "tools"),
    ],
)
@pytest.mark.asyncio
async def test_auto_uses_profile_capability_and_actual_agent_mode(profile: dict[str, Any], mode: str) -> None:
    model, calls = recording_model(profile=profile)
    session = session_for(model)
    result = await session.ask(Query(tail="answer", output=Answer))
    assert session.output_mode == "auto"
    assert session.resolved_output_mode == mode
    assert result.output == Answer(value=7)
    assert result.resolved_output_mode == mode
    assert result.failure is None
    assert len(calls) == 1
    assert [tool.name for tool in calls[0].output_tools] == (["Answer", "Note"] if mode == "tools" else [])


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.asyncio
async def test_testmodel_receives_explicit_sdk_output_mode(mode: str) -> None:
    profile = ModelProfile(supports_forced_tool_choice=mode == "tools")
    model = TestModel(
        profile=profile,
        custom_output_args={"value": 3} if mode == "tools" else None,
        custom_output_text='{"value":3}' if mode == "prompted" else None,
    )
    session = session_for(model)
    result = await session.ask(Query(tail="answer", output=Answer))
    assert result.output == Answer(value=3)
    params = model.last_model_request_parameters
    assert params is not None
    assert params.output_mode == ("tool" if mode == "tools" else "prompted")
    assert bool(params.output_tools) == (mode == "tools")


@pytest.mark.parametrize(
    ("profile", "settings", "mode"),
    [
        ({"supports_thinking": True}, {"thinking": True}, "prompted"),
        ({"supports_thinking": True}, {"thinking": "high"}, "prompted"),
        ({"supports_thinking": True, "thinking_enabled_by_default": True}, {"thinking": False}, "tools"),
        ({"thinking_enabled_by_default": True}, {"thinking": False}, "prompted"),
        ({}, {"thinking": True}, "tools"),  # SDK ignores unsupported unified controls.
        ({"supports_thinking": True, "thinking_always_enabled": True}, {"thinking": False}, "prompted"),
        ({}, {"openai_reasoning_effort": "high"}, "prompted"),
        ({"supports_thinking": True}, {"thinking": True, "openai_reasoning_effort": "none"}, "tools"),
        ({}, {"extra_body": {"thinking": {"type": "enabled"}}}, "prompted"),
        ({}, {"extra_body": {"thinking": {"type": "adaptive"}}}, "prompted"),
        ({"thinking_enabled_by_default": True}, {"extra_body": {"thinking": {"type": "disabled"}}}, "tools"),
        ({}, {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}, "prompted"),
        ({}, {"chat_template_kwargs": {"enable_thinking": True}}, "prompted"),
        ({}, {"extra_body": {"reasoning_effort": "low"}}, "prompted"),
        ({}, {"anthropic_thinking": {"type": "adaptive"}}, "prompted"),
        ({"thinking_enabled_by_default": True}, {"anthropic_thinking": {"type": "disabled"}}, "tools"),
        ({}, {"google_thinking_config": {"thinking_budget": -1}}, "prompted"),
        ({}, {"google_thinking_config": {"thinking_budget": 0}}, "tools"),
        ({}, {"google_thinking_config": {"thinking_level": "MINIMAL"}}, "prompted"),
        ({}, {"google_thinking_config": {"include_thoughts": True}}, "tools"),
        ({}, {"openai_reasoning_effort": "none", "extra_body": {"thinking": {"type": "enabled"}}}, "prompted"),
    ],
)
def test_thinking_controls_and_provider_precedence(
    profile: dict[str, Any], settings: dict[str, Any], mode: str
) -> None:
    model, _ = recording_model(profile={"supports_forced_tool_choice_with_thinking": False, **profile})
    assert session_for(model, model_settings=settings).resolved_output_mode == mode


@pytest.mark.asyncio
async def test_three_settings_layers_merge_shallowly_and_same_mode_query_runs() -> None:
    model, calls = recording_model(
        profile={"supports_thinking": True, "supports_forced_tool_choice_with_thinking": False},
        settings={"thinking": True, "temperature": 0.2, "extra_body": {"thinking": {"type": "enabled"}}},
    )
    inherited = session_for(model)
    assert inherited.resolved_output_mode == "prompted"
    session = session_for(model, model_settings={"thinking": False, "extra_body": {}, "temperature": 0.4})
    assert session.resolved_output_mode == "tools"
    result = await session.ask(Query(tail="answer", output=Answer, model_settings={"temperature": 0.8}))
    assert result.output == Answer(value=7)
    assert calls[0].model_settings["temperature"] == 0.8
    assert calls[0].model_request_parameters.thinking is False
    assert calls[0].model_settings["extra_body"] == {}


@pytest.mark.parametrize("entrypoint", ["ask", "run_once", "fork"])
@pytest.mark.parametrize("initial_thinking", [False, True])
@pytest.mark.asyncio
async def test_auto_query_flip_rejected_before_budget_and_provider(entrypoint: str, initial_thinking: bool) -> None:
    model, calls = recording_model(
        profile={"supports_thinking": True, "supports_forced_tool_choice_with_thinking": False}
    )
    session = session_for(
        model,
        model_settings={"thinking": initial_thinking},
        policy=Policy(retries=0, budget=Budget(max_requests=1)),
    )
    query = Query(tail="flip", output=Answer, id="preassigned", model_settings={"thinking": not initial_thinking})
    with pytest.raises(OutputModeConflictError) as error:
        if entrypoint == "fork":
            await session.fork([Query(tail="valid", output=Answer), query])
        else:
            await getattr(session, entrypoint)(query)
    assert error.value.code == "output_mode_conflict"
    assert calls == []
    assert session._budget.requests == 0
    result = await session.ask(Query(tail="unchanged", output=Answer))
    assert result.output == Answer(value=7)
    assert session._budget.requests == 1


@pytest.mark.parametrize("policy", ["warn", "error", "ignore"])
def test_explicit_unavailable_tools_build_policy_does_not_change_mode(policy: str) -> None:
    model, calls = recording_model(profile={"supports_forced_tool_choice": False})
    if policy == "error":
        with pytest.raises(UnforcedToolsError) as error:
            session_for(model, output_mode="tools", on_unforced_tools=policy)
        assert error.value.code == "unforced_tools"
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            session = session_for(model, output_mode="tools", on_unforced_tools=policy)
        assert session.output_mode == session.resolved_output_mode == "tools"
        assert [item.category for item in caught] == ([UnforcedToolsWarning] if policy == "warn" else [])
    assert calls == []


@pytest.mark.parametrize("policy", ["warn", "error", "ignore"])
@pytest.mark.asyncio
async def test_explicit_tools_query_capability_change_policy(policy: str) -> None:
    model, calls = recording_model(
        profile={"supports_thinking": True, "supports_forced_tool_choice_with_thinking": False}
    )
    session = session_for(
        model, output_mode="tools", on_unforced_tools=policy, policy=Policy(budget=Budget(max_requests=1))
    )
    query = Query(tail="thinking", output=Answer, model_settings={"thinking": True})
    if policy == "error":
        with pytest.raises(UnforcedToolsError):
            await session.ask(query)
        assert calls == []
        assert session._budget.requests == 0
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await session.ask(query)
        assert result.resolved_output_mode == "tools"
        assert [item.category for item in caught] == ([UnforcedToolsWarning] if policy == "warn" else [])
        assert len(calls) == 1


@pytest.mark.parametrize("settings", [{"vendor_thinking": True}, {"extra_body": {"reasoning": "mystery"}}])
def test_unknown_thinking_control_requires_explicit_mode(settings: dict[str, Any]) -> None:
    model, _ = recording_model(profile={"supports_forced_tool_choice_with_thinking": False})
    with pytest.raises(ForkReduceError) as error:
        session_for(model, model_settings=settings)
    assert error.value.code == "unknown_thinking_control"
    assert session_for(model, output_mode="prompted", model_settings=settings).resolved_output_mode == "prompted"
    with pytest.warns(UnforcedToolsWarning):
        explicit = session_for(model, output_mode="tools", model_settings=settings)
    assert explicit.resolved_output_mode == "tools"


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("payload", [{"value": "not-an-integer"}, {"value": 1, "unexpected": True}])
@pytest.mark.asyncio
async def test_explicit_modes_validate_output_without_fallback(mode: str, payload: dict[str, Any]) -> None:
    def invalid(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        part = ToolCallPart("Answer", payload) if mode == "tools" else TextPart(json.dumps(payload))
        return ModelResponse(parts=[part])

    session = session_for(FunctionModel(invalid), output_mode=mode)
    result = await session.ask(Query(tail="invalid", output=Answer))
    assert result.output is None
    assert result.failure is not None
    assert result.resolved_output_mode == mode
    assert result.attempts == 1


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.asyncio
async def test_explicit_modes_do_not_flip_when_query_changes_thinking(mode: str) -> None:
    model, _ = recording_model(
        profile={"supports_thinking": True, "supports_forced_tool_choice_with_thinking": False},
        settings={"thinking": True},
    )
    session = session_for(model, output_mode=mode, on_unforced_tools="ignore")
    result = await session.ask(Query(tail="answer", output=Answer, model_settings={"thinking": False}))
    assert result.output == Answer(value=7)
    assert result.resolved_output_mode == mode


@pytest.mark.parametrize("kind", ["transport", "wrong_output", "budget"])
@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.asyncio
async def test_all_failure_paths_record_actual_mode(kind: str, mode: str) -> None:
    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if kind == "transport":
            raise ConnectionError("offline connection failure")
        if mode == "tools":
            return ModelResponse(parts=[ToolCallPart("Note", {"text": "different registered type"})])
        return ModelResponse(parts=[TextPart('{"value":"invalid"}')])

    session = session_for(
        FunctionModel(respond),
        output_mode=mode,
        policy=Policy(retries=0, budget=Budget(max_requests=0) if kind == "budget" else None),
    )
    result = await session.ask(Query(tail="failure", output=Answer))
    assert result.failure is not None
    assert result.resolved_output_mode == mode
    assert result.attempts == (0 if kind == "budget" else 1)


@pytest.mark.parametrize("mode", ["tools", "prompted"])
def test_historical_011_canonical_hash_uses_actual_mode_without_auto_marker(mode: str) -> None:
    model, _ = recording_model(profile={"supports_forced_tool_choice": mode == "tools"})
    auto = session_for(model)
    explicit = session_for(model, output_mode=mode)
    # Independent transcription of the 0.1.1 canonical payload and encoding.
    payload = {
        "version": 1,
        "output_mode": mode,
        "instructions": "Return grounded answers.",
        "documents": coerce_documents("α shared evidence").rendered,
        "tools": [
            {"name": "Answer", "description": "A structured answer.", "schema": Answer.model_json_schema()},
            {"name": "Note", "description": "A note.", "schema": Note.model_json_schema()},
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    expected = hashlib.sha256(canonical).hexdigest()
    assert auto.prefix_hash == explicit.prefix_hash == expected


@pytest.mark.parametrize("value", ["native", "tool", "AUTO", None, [], 1])
def test_unknown_output_modes_are_caller_errors(value: Any) -> None:
    with pytest.raises(ForkReduceError, match="output_mode must"):
        session_for(TestModel(), output_mode=value)


@pytest.mark.parametrize("value", ["warning", "WARN", None, [], 1])
def test_unknown_unforced_tools_policies_are_caller_errors(value: Any) -> None:
    with pytest.raises(ForkReduceError, match="on_unforced_tools must"):
        session_for(TestModel(), on_unforced_tools=value)


@pytest.mark.parametrize(
    "override",
    [
        {"openai_reasoning_effort": "high"},
        {"anthropic_thinking": {"type": "adaptive"}},
        {"google_thinking_config": {"thinking_budget": 100}},
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    ],
)
@pytest.mark.asyncio
async def test_query_provider_controls_cannot_change_frozen_auto_mode(override: dict[str, Any]) -> None:
    model, calls = recording_model(profile={"supports_forced_tool_choice_with_thinking": False})
    session = session_for(model, policy=Policy(budget=Budget(max_requests=1)))
    with pytest.raises(OutputModeConflictError):
        await session.ask(Query(tail="provider override", output=Answer, model_settings=override))
    assert calls == []
    assert session._budget.requests == 0


@pytest.mark.asyncio
async def test_query_unknown_thinking_control_fails_before_budget() -> None:
    model, calls = recording_model(profile={"supports_forced_tool_choice_with_thinking": False})
    session = session_for(model, policy=Policy(budget=Budget(max_requests=1)))
    with pytest.raises(ForkReduceError) as error:
        await session.ask(Query(tail="unknown", output=Answer, model_settings={"vendor_thinking": True}))
    assert error.value.code == "unknown_thinking_control"
    assert calls == []
    assert session._budget.requests == 0


@pytest.mark.asyncio
async def test_tools_mode_does_not_fallback_to_json_text() -> None:
    def text_only(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('{"value":7}')])

    session = session_for(FunctionModel(text_only))
    result = await session.ask(Query(tail="text instead of tool", output=Answer))
    assert result.output is None
    assert result.failure is not None
    assert result.resolved_output_mode == "tools"


@pytest.mark.asyncio
async def test_mutated_model_defaults_cannot_change_frozen_auto_mode() -> None:
    model, calls = recording_model(
        profile={"supports_thinking": True, "supports_forced_tool_choice_with_thinking": False},
        settings={"thinking": False},
    )
    session = session_for(model)
    assert model.settings is not None
    model.settings["thinking"] = True
    with pytest.raises(OutputModeConflictError):
        await session.run_once(Query(tail="changed defaults", output=Answer, id="already-has-id"))
    assert calls == []


@pytest.mark.parametrize(
    ("profile", "mode"),
    [
        ({}, "tools"),
        ({"supports_forced_tool_choice_with_thinking": True}, "tools"),
        ({"supports_tools": False}, "prompted"),
        ({"supports_forced_tool_choice": False}, "prompted"),
        ({"thinking_always_enabled": True, "supports_forced_tool_choice_with_thinking": False}, "prompted"),
    ],
)
@pytest.mark.parametrize("control", ["vendor_thinking", "openai_send_reasoning_ids"])
@pytest.mark.asyncio
async def test_irrelevant_unknown_controls_do_not_block_unconditional_modes(
    profile: dict[str, Any], mode: str, control: str
) -> None:
    model, calls = recording_model(profile=profile)
    session = session_for(model, model_settings={control: True})
    assert session.resolved_output_mode == mode
    result = await session.ask(Query(tail="unknown but irrelevant", output=Answer, model_settings={control: False}))
    assert result.output == Answer(value=7)
    assert result.resolved_output_mode == mode
    assert len(calls) == 1
