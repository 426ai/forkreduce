"""Offline dynamic JSON Schema outputs with a fixed provider prefix."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any, cast

import pytest
from pydantic import BaseModel, create_model
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage
from tests.test_output_feedback import feedback_text, prefix_bytes

from forkreduce import (
    Budget,
    FailureKind,
    ForkReduceError,
    OutputModeConflictError,
    OutputSpec,
    Policy,
    Query,
    UnregisteredOutputError,
    build,
)
from forkreduce.output_specs import DYNAMIC_OUTPUT_TOOL_NAME
from forkreduce.session import OutputValidator, Session


class FixedAnswer(BaseModel):
    """A fixed output available alongside dynamic ones."""

    value: int


class FixedNote(BaseModel):
    """A second fixed output in the complete frozen tool list."""

    text: str


class DynamicModel:
    def __init__(self, payload: Callable[[int], dict[str, Any]], *, profile: ModelProfile | None = None) -> None:
        self.payload = payload
        self.tool_name = DYNAMIC_OUTPUT_TOOL_NAME
        self.calls: list[tuple[list[ModelMessage], AgentInfo]] = []
        self.model = FunctionModel(self.respond, profile=profile)

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(info)))
        value = self.payload(len(self.calls))
        part = ToolCallPart(self.tool_name, value) if info.output_tools else TextPart(json.dumps(value))
        return ModelResponse(parts=[part], usage=RequestUsage(input_tokens=10, output_tokens=3, cache_read_tokens=4))


def object_spec(constraint: dict[str, Any], *, strict: bool = True, name: str = "DynamicAnswer") -> OutputSpec:
    return OutputSpec(
        name,
        {
            "type": "object",
            "properties": {"value": constraint},
            "required": ["value"],
            "additionalProperties": False,
        },
        description="Return the requested synthetic value.",
        strict=strict,
    )


def dynamic_session(
    recorded: DynamicModel,
    mode: str,
    *,
    feedback: int = 0,
    validator: OutputValidator | None = None,
    fixed: bool = False,
    policy: Policy | None = None,
) -> Session:
    return build(
        recorded.model,
        documents="α shared synthetic evidence",
        instructions="Return a grounded synthetic JSON answer.",
        outputs=[FixedAnswer, FixedNote] if fixed else (),
        output_mode=mode,  # type: ignore[arg-type]
        dynamic_outputs="generic_tool" if mode == "tools" else "prompted",
        warmup=False,
        output_validator=validator,
        policy=policy or Policy(retries=0, retry_backoff=0, output_feedback_retries=feedback),
    )


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("fixed", [False, True])
async def test_different_query_schemas_share_hash_prefix_and_complete_tools(mode: str, fixed: bool) -> None:
    recorded = DynamicModel(lambda n: {"value": "synthetic" if n == 1 else 7})
    session = dynamic_session(recorded, mode, fixed=fixed)
    specs = [object_spec({"type": "string"}, name="Label"), object_spec({"type": "integer"}, name="Count")]
    frozen_hash, frozen_tools = session.prefix_hash, copy.deepcopy(session.tool_specs)
    results = [await session.ask(Query[dict[str, Any]](tail="Answer.", output=spec)) for spec in specs]
    assert [result.output for result in results] == [{"value": "synthetic"}, {"value": 7}]
    assert all(result.failure is None and result.prefix_hash == frozen_hash for result in results)
    assert session.prefix_hash == frozen_hash
    assert session.tool_specs == frozen_tools
    assert session.output_names == (["FixedAnswer", "FixedNote"] if fixed else [])
    first_messages, first_info = recorded.calls[0]
    second_messages, second_info = recorded.calls[1]
    assert prefix_bytes(first_messages) == prefix_bytes(second_messages) == session.rendered_documents.encode("utf-8")
    assert first_info.output_tools == second_info.output_tools
    assert first_info.instructions == second_info.instructions == session.instructions
    assert "properties" not in (first_info.instructions or "")
    expected = (["FixedAnswer", "FixedNote"] if fixed else []) + [DYNAMIC_OUTPUT_TOOL_NAME]
    assert [tool.name for tool in first_info.output_tools] == (expected if mode == "tools" else [])
    if mode == "tools":
        generic = first_info.output_tools[-1]
        assert generic.parameters_json_schema["type"] == "object"
        assert generic.parameters_json_schema["additionalProperties"] is True
        assert "value" not in generic.parameters_json_schema.get("properties", {})
        assert generic.strict is False
    for spec in specs:
        assert json.dumps(spec.json_schema, sort_keys=True, ensure_ascii=False) in session.render_tail(
            Query[dict[str, Any]](tail="Answer.", output=spec)
        )
    assert all(session.starts_with_prefix(result.messages) for result in results)


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("strict", [False, True])
async def test_schema_failure_uses_native_feedback_and_request_accounting(mode: str, strict: bool) -> None:
    recorded = DynamicModel(lambda n: {"value": -1 if n == 1 else 7})
    session = dynamic_session(recorded, mode, feedback=1, fixed=True)
    baseline = dynamic_session(recorded, mode, fixed=True)
    result = await session.ask(Query[dict[str, Any]](tail="Answer.", output=object_spec({"minimum": 1}, strict=strict)))
    assert result.output == {"value": 7}
    assert result.failure is None
    assert result.attempts == 1
    assert result.feedback_retries_used == result.meta["feedback_retries_used"] == 1
    assert result.usage.requests == len(recorded.calls) == 2
    assert (result.usage.input_tokens, result.usage.output_tokens, result.cache_hit_tokens) == (20, 6, 8)
    assert (session._budget.input_tokens, session._budget.output_tokens) == (20, 6)
    assert "minimum" in feedback_text(recorded.calls[1][0])
    assert prefix_bytes(recorded.calls[0][0]) == prefix_bytes(recorded.calls[1][0])
    assert recorded.calls[0][1].output_tools == recorded.calls[1][1].output_tools
    assert recorded.calls[0][1].instructions == recorded.calls[1][1].instructions == session.instructions
    assert session.prefix_hash == baseline.prefix_hash
    assert session.tool_specs == baseline.tool_specs


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("constraint", "value"),
    [
        ({"type": "integer"}, "7"),
        ({"type": "integer"}, True),
        ({"type": "boolean"}, 1),
        ({"type": "string"}, 7),
        ({"type": "number", "maximum": 3}, 4),
        ({"type": "integer", "multipleOf": 2}, 3),
        ({"type": "string", "minLength": 3}, "x"),
        ({"type": "string", "pattern": "^a"}, "bbb"),
        ({"const": "expected"}, "other"),
        ({"enum": ["a", "b"]}, "c"),
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "no"),
        ({"type": "array", "items": {"type": "integer"}}, ["7"]),
        ({"type": "object", "required": ["child"]}, {}),
    ],
)
async def test_both_strict_modes_always_validate_without_coercion(
    mode: str, strict: bool, constraint: dict[str, Any], value: Any
) -> None:
    recorded = DynamicModel(lambda _n: {"value": value})
    result = await dynamic_session(recorded, mode).ask(
        Query[dict[str, Any]](tail="Answer.", output=object_spec(constraint, strict=strict))
    )
    assert result.output is None
    assert result.failure is not None and result.failure.kind is FailureKind.VALIDATION
    assert result.feedback_retries_used == 0
    assert len(recorded.calls) == result.usage.requests == 1


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("payload", [{}, {"value": 7, "unexpected": True}])
async def test_required_fields_and_extra_properties_always_validate(
    mode: str, strict: bool, payload: dict[str, Any]
) -> None:
    recorded = DynamicModel(lambda _n: payload)
    result = await dynamic_session(recorded, mode).ask(
        Query[dict[str, Any]](tail="Answer.", output=object_spec({"type": "integer"}, strict=strict))
    )
    assert result.failure is not None and result.failure.kind is FailureKind.VALIDATION
    assert len(recorded.calls) == 1


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("strict", [False, True])
async def test_local_refs_defs_anyof_and_recursion_are_not_rewritten(mode: str, strict: bool) -> None:
    schema = {
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "value": {"const": "node"},
                    "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]},
                },
                "required": ["value", "next"],
                "additionalProperties": False,
            }
        },
        "$ref": "#/$defs/node",
    }
    before = copy.deepcopy(schema)
    spec = OutputSpec("Linked", schema, strict=strict)
    value = {"value": "node", "next": {"value": "node", "next": None}}
    recorded = DynamicModel(lambda _n: value)
    session = dynamic_session(recorded, mode)
    result = await session.ask(Query[dict[str, Any]](tail="Return nodes.", output=spec))
    assert result.output == value and result.failure is None
    assert schema == spec.json_schema == before
    assert json.dumps(before, sort_keys=True) in session.render_tail(Query[dict[str, Any]](tail="Nodes.", output=spec))
    invalid = DynamicModel(lambda _n: {"value": "node", "next": {"value": "wrong", "next": None}})
    rejected = await dynamic_session(invalid, mode).ask(Query[dict[str, Any]](tail="Nodes.", output=spec))
    assert rejected.failure is not None and rejected.failure.kind is FailureKind.VALIDATION


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("format_name,value", [("email", "not-an-email"), ("date", "not-a-date")])
async def test_strict_enables_optional_format_checks(mode: str, format_name: str, value: str) -> None:
    for strict in (True, False):
        recorded = DynamicModel(lambda _n: {"value": value})
        spec = object_spec({"type": "string", "format": format_name}, strict=strict)
        result = await dynamic_session(recorded, mode).ask(Query[dict[str, Any]](tail="Answer.", output=spec))
        if strict:
            assert result.failure is not None and result.failure.kind is FailureKind.VALIDATION
        else:
            assert result.failure is None and result.output == {"value": value}


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
async def test_strict_rejects_nonfinite_numbers_even_in_unconstrained_nested_values(mode: str, number: float) -> None:
    recorded = DynamicModel(lambda _n: {"value": {"nested": [number]}})
    result = await dynamic_session(recorded, mode).ask(
        Query[dict[str, Any]](tail="Answer.", output=object_spec({}, strict=True))
    )
    assert result.failure is not None and result.failure.kind is FailureKind.VALIDATION
    assert len(recorded.calls) == 1


def positive(output: Any) -> Any:
    assert isinstance(output, dict)
    if output["value"] <= 0:
        raise ModelRetry("application requires a positive value")
    return output


async def positive_async(output: Any) -> Any:
    return positive(output)


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("validator", [positive, positive_async])
@pytest.mark.parametrize("feedback", [0, 1])
async def test_schema_then_sync_or_async_application_validator(
    mode: str, validator: OutputValidator, feedback: int
) -> None:
    recorded = DynamicModel(lambda n: {"value": 0 if n == 1 else 8})
    session = dynamic_session(recorded, mode, feedback=feedback, validator=validator)
    result = await session.ask(Query[dict[str, Any]](tail="Answer.", output=object_spec({"type": "integer"})))
    assert result.feedback_retries_used == feedback
    assert result.usage.requests == len(recorded.calls) == feedback + 1
    if feedback:
        assert result.output == {"value": 8} and result.failure is None
        assert "application requires a positive value" in feedback_text(recorded.calls[1][0])
    else:
        assert result.failure is not None and result.failure.kind is FailureKind.VALIDATION


@pytest.mark.parametrize("mode", ["tools", "prompted"])
async def test_application_validator_only_receives_schema_valid_values(mode: str) -> None:
    seen: list[Any] = []

    def validate(output: Any) -> Any:
        seen.append(output)
        return output

    recorded = DynamicModel(lambda n: {"value": "not-integer" if n == 1 else 4})
    result = await dynamic_session(recorded, mode, feedback=1, validator=validate).ask(
        Query[dict[str, Any]](tail="Answer.", output=object_spec({"type": "integer"}))
    )
    assert result.output == {"value": 4} and result.failure is None
    assert seen == [{"value": 4}]


@pytest.mark.parametrize("mode", ["tools", "prompted"])
async def test_dynamic_feedback_exhaustion_accounts_all_requests(mode: str) -> None:
    recorded = DynamicModel(lambda _n: {"value": "bad"})
    result = await dynamic_session(recorded, mode, feedback=2).ask(
        Query[dict[str, Any]](tail="Answer.", output=object_spec({"type": "integer"}))
    )
    assert result.failure is not None and result.failure.kind is FailureKind.VALIDATION
    assert result.feedback_retries_used == 2
    assert result.usage.requests == len(recorded.calls) == 3


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("budget", [Budget(max_requests=1), Budget(max_input_tokens=10), Budget(max_output_tokens=3)])
async def test_dynamic_feedback_obeys_request_and_token_budgets(mode: str, budget: Budget) -> None:
    recorded = DynamicModel(lambda _n: {"value": "bad"})
    policy = Policy(retries=0, output_feedback_retries=2, budget=budget)
    session = dynamic_session(recorded, mode, policy=policy)
    result = await session.ask(Query[dict[str, Any]](tail="Answer.", output=object_spec({"type": "integer"})))
    assert result.failure is not None and result.failure.kind is FailureKind.BUDGET
    assert result.feedback_retries_used == 0
    assert result.usage.requests == len(recorded.calls) == session._budget.requests == 1
    assert (session._budget.input_tokens, session._budget.output_tokens) == (10, 3)


@pytest.mark.parametrize(
    ("outputs", "dynamic", "profile", "expected"),
    [
        ((), None, {}, "prompted"),
        ((), "prompted", {}, "prompted"),
        ((), "generic_tool", {}, "tools"),
        ((), "generic_tool", {"supports_forced_tool_choice": False}, "prompted"),
        ((FixedAnswer,), None, {}, "tools"),
        ((FixedAnswer,), "prompted", {}, "prompted"),
        ((FixedAnswer,), "generic_tool", {}, "tools"),
        ((FixedAnswer,), "generic_tool", {"supports_tools": False}, "prompted"),
    ],
)
def test_auto_dynamic_rules_and_legacy_fixed_selection(
    outputs: tuple[type[BaseModel], ...], dynamic: str | None, profile: dict[str, Any], expected: str
) -> None:
    recorded = DynamicModel(lambda _n: {"value": 4}, profile=cast(ModelProfile, profile))
    session = build(
        recorded.model,
        outputs=outputs,
        documents="Synthetic evidence.",
        instructions="Answer.",
        dynamic_outputs=dynamic,  # type: ignore[arg-type]
    )
    assert session.resolved_output_mode == expected
    assert session.dynamic_outputs == dynamic
    assert len(session.tool_specs) == len(outputs) + int(dynamic == "generic_tool")


@pytest.mark.parametrize("entrypoint", ["ask", "fork", "run_once", "render_tail"])
async def test_dynamic_query_on_static_tools_rejects_before_any_request(entrypoint: str) -> None:
    recorded = DynamicModel(lambda _n: {"value": 4})
    session = build(recorded.model, outputs=[FixedAnswer], documents="Synthetic", instructions="Answer.")
    query = Query[dict[str, Any]](tail="Answer.", output=object_spec({"type": "integer"}))
    frozen_hash, frozen_tools = session.prefix_hash, copy.deepcopy(session.tool_specs)
    with pytest.raises(UnregisteredOutputError, match="dynamic_outputs='generic_tool'.*prompted"):
        if entrypoint == "fork":
            await session.fork([query])
        elif entrypoint == "render_tail":
            session.render_tail(query)
        else:
            await getattr(session, entrypoint)(query)
    assert not recorded.calls
    assert session._budget.requests == 0
    assert session.prefix_hash == frozen_hash and session.tool_specs == frozen_tools


@pytest.mark.parametrize("mode", ["tools", "prompted"])
async def test_generic_registration_preserves_tools_for_fixed_queries(mode: str) -> None:
    recorded = DynamicModel(lambda _n: {"value": 9})
    session = dynamic_session(recorded, mode, fixed=True)
    dynamic = await session.ask(Query[dict[str, Any]](tail="Dynamic.", output=object_spec({"type": "integer"})))
    recorded.tool_name = "FixedAnswer"
    fixed = await session.ask(Query(tail="Fixed.", output=FixedAnswer))
    assert dynamic.output == {"value": 9}
    assert fixed.output == FixedAnswer(value=9)
    assert fixed.failure is None
    assert recorded.calls[0][1].output_tools == recorded.calls[1][1].output_tools
    assert recorded.calls[0][1].instructions == recorded.calls[1][1].instructions == session.instructions


@pytest.mark.parametrize("mode", ["tools", "prompted"])
def test_generic_registration_changes_hash_but_query_specs_never_do(mode: str) -> None:
    recorded = DynamicModel(lambda _n: {"value": 4})
    base = build(
        recorded.model,
        outputs=[FixedAnswer],
        documents="Synthetic",
        instructions="Answer.",
        output_mode=mode,  # type: ignore[arg-type]
    )
    generic = build(
        recorded.model,
        outputs=[FixedAnswer],
        documents="Synthetic",
        instructions="Answer.",
        output_mode=mode,
        dynamic_outputs="generic_tool",  # type: ignore[arg-type]
    )
    assert base.prefix_hash != generic.prefix_hash
    assert base.tool_specs == generic.tool_specs[:-1]
    before = generic.prefix_hash
    for spec in (object_spec({"const": 1}), object_spec({"const": "different"}, strict=False)):
        generic.prepare_query(Query[dict[str, Any]](tail="Answer.", output=spec))
    assert generic.prefix_hash == before


def test_explicit_dynamic_tools_require_generic_and_reject_name_collision() -> None:
    recorded = DynamicModel(lambda _n: {})
    for dynamic in (None, "prompted"):
        with pytest.raises(ForkReduceError, match="requires dynamic_outputs='generic_tool'"):
            build(
                recorded.model,
                documents="Synthetic",
                instructions="Answer.",
                output_mode="tools",
                dynamic_outputs=dynamic,  # type: ignore[arg-type]
            )
    colliding = create_model(DYNAMIC_OUTPUT_TOOL_NAME, value=(int, ...))
    with pytest.raises(ForkReduceError, match="reserved"):
        build(
            recorded.model,
            outputs=[colliding],
            documents="Synthetic",
            instructions="Answer.",
            dynamic_outputs="generic_tool",
        )
    with pytest.raises(ForkReduceError, match="dynamic_outputs must be"):
        build(
            recorded.model,
            documents="Synthetic",
            instructions="Answer.",
            dynamic_outputs="unsupported",  # type: ignore[arg-type]
        )


def test_schema_copy_isolation_and_serializable_dataclass_fields() -> None:
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}
    original = copy.deepcopy(schema)
    spec = OutputSpec("Answer", schema, description="Synthetic", strict=False)
    schema["properties"]["value"]["type"] = "string"  # type: ignore[index]
    schema["required"].append("changed")  # type: ignore[union-attr]
    assert spec.json_schema == original
    encoded = json.loads(json.dumps(asdict(spec), allow_nan=False))
    rebuilt = OutputSpec(**encoded)
    assert rebuilt == spec and rebuilt.json_schema is not spec.json_schema
    cloned = replace(spec)
    cloned.json_schema["required"].append("independent")
    assert spec.json_schema == original


@pytest.mark.parametrize(
    "reference", ["https://example.invalid/schema", "file:///tmp/schema", "other.json#/$defs/value"]
)
@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef", "$recursiveRef"])
def test_remote_references_rejected_before_any_resolution(
    reference: str, keyword: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("remote reference must be rejected before validator construction or resolution")

    monkeypatch.setattr("jsonschema.validators.validator_for", forbidden)
    with pytest.raises(ValueError, match="only supports local references"):
        OutputSpec("Remote", {"type": "object", "$defs": {"unused": {keyword: reference}}})


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "not-a-json-schema-type"},
        {"type": "object", "required": "not-a-list"},
        {"const": float("nan")},
        {"const": (1, 2)},
        {1: "nonstring-key"},
    ],
)
def test_invalid_or_nonserializable_schemas_rejected(schema: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        OutputSpec("Invalid", schema)


@pytest.mark.parametrize("name", ["", "   "])
def test_output_name_must_be_nonempty(name: str) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        OutputSpec(name, {"type": "object"})


@pytest.mark.parametrize("fixed", [False, True])
async def test_dynamic_schema_runs_on_default_or_explicit_prompted_session_without_generic(fixed: bool) -> None:
    recorded = DynamicModel(lambda _n: {"value": 4})
    session = build(
        recorded.model,
        documents="Synthetic evidence.",
        instructions="Answer.",
        outputs=[FixedAnswer] if fixed else (),
        output_mode="prompted" if fixed else "auto",
        policy=Policy(retries=0),
    )
    assert session.dynamic_outputs is None
    assert session.resolved_output_mode == "prompted"
    result = await session.ask(Query[dict[str, Any]](tail="Answer.", output=object_spec({"type": "integer"})))
    assert result.output == {"value": 4} and result.failure is None
    assert type(result.output) is dict
    assert not recorded.calls[0][1].output_tools


@pytest.mark.parametrize("mode", ["tools", "prompted"])
async def test_reference_like_literal_data_is_not_treated_as_a_schema_reference(mode: str) -> None:
    value = {"$ref": "https://example.invalid/literal", "const": {"$ref": "literal.json"}}
    schema = {"type": "object", "const": value, "properties": {"$ref": {"type": "string"}}}
    spec = OutputSpec("Literal", schema)
    recorded = DynamicModel(lambda _n: value)
    result = await dynamic_session(recorded, mode).ask(Query[dict[str, Any]](tail="Answer.", output=spec))
    assert result.output == value and result.failure is None


def test_circular_schema_data_is_not_serializable() -> None:
    schema: dict[str, Any] = {"type": "object"}
    schema["properties"] = {"cycle": schema}
    with pytest.raises(ValueError, match="serializable JSON"):
        OutputSpec("Circular", schema)


def test_generic_auto_conservatively_prompts_for_unknown_thinking_control() -> None:
    recorded = DynamicModel(
        lambda _n: {"value": 4},
        profile=ModelProfile(supports_forced_tool_choice_with_thinking=False),
    )
    session = build(
        recorded.model,
        documents="Synthetic",
        instructions="Answer.",
        dynamic_outputs="generic_tool",
        model_settings={"extra_body": {"thinking_unknown": True}},
    )
    assert session.resolved_output_mode == "prompted"


async def test_generic_auto_query_cannot_change_frozen_output_mode() -> None:
    recorded = DynamicModel(
        lambda _n: {"value": 4},
        profile=ModelProfile(supports_thinking=True, supports_forced_tool_choice_with_thinking=False),
    )
    session = build(
        recorded.model,
        documents="Synthetic",
        instructions="Answer.",
        dynamic_outputs="generic_tool",
    )
    assert session.resolved_output_mode == "tools"
    query = Query[dict[str, Any]](
        tail="Answer.",
        output=object_spec({"type": "integer"}),
        model_settings={"thinking": True},
    )
    with pytest.raises(OutputModeConflictError, match="frozen output mode"):
        await session.ask(query)
    assert not recorded.calls and session._budget.requests == 0
