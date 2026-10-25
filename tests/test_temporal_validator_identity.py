"""Worker-local callback identity is stricter than serialized callable names."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.support import Scripted, tool_result

from forkreduce import OutputValidator, Policy, Session, build
from forkreduce.contrib import temporal as adapter
from forkreduce.errors import ForkReduceError


class IdentityAnswer(BaseModel):
    text: str


class StableValidator:
    def __init__(self, label: str, observed: list[str], *, named: bool = False) -> None:
        self.label = label
        self.observed = observed
        if named:
            self.__qualname__ = "stable_validator"

    def __repr__(self) -> str:
        return "StableValidator()"

    def __call__(self, output: IdentityAnswer) -> IdentityAnswer:
        self.observed.append(self.label)
        return IdentityAnswer(text=f"{output.text}:{self.label}")


def make_validator(kind: str, label: str, observed: list[str]) -> Callable[[IdentityAnswer], IdentityAnswer]:
    if kind == "closure":

        def validate(output: IdentityAnswer) -> IdentityAnswer:
            observed.append(label)
            return IdentityAnswer(text=f"{output.text}:{label}")

        return validate
    return StableValidator(label, observed, named=kind == "named_instance")


def make_session(model: FunctionModel, validator: OutputValidator | None) -> Session:
    return build(
        model,
        documents="Synthetic callback identity evidence.",
        instructions="Return an answer.",
        outputs=[IdentityAnswer],
        output_validator=validator,
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )


def respond(_messages: list[ModelMessage], info: AgentInfo, _n: int) -> ModelResponse:
    return tool_result(info, "IdentityAnswer", {"text": "raw"})


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_SESSIONS", "_FINGERPRINTS", "_FACTORIES"):
        monkeypatch.setattr(adapter, name, {})
    monkeypatch.setattr(adapter, "_STORE", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["closure", "repr_instance", "named_instance"])
async def test_colliding_validators_cannot_replace_registered_session(kind: str) -> None:
    observed: list[str] = []
    model = Scripted(respond).model()
    first_validator = make_validator(kind, "first", observed)
    other_validator = make_validator(kind, "other", observed)
    first = make_session(model, first_validator)
    other = make_session(model, other_validator)
    assert first_validator is not other_validator
    assert first.prefix_hash == other.prefix_hash
    assert adapter.execution_fingerprint(first) == adapter.execution_fingerprint(other)

    prefix_hash = adapter.register_session(first)
    assert adapter.register_session(first) == prefix_hash
    with pytest.raises(ForkReduceError, match="different output validator"):
        adapter.register_session(other)
    assert adapter._SESSIONS[prefix_hash] is first

    output = await adapter.execute_query_activity(
        adapter.query_activity_input(first, query_id="first", tail="Question", output_name="IdentityAnswer")
    )
    assert output.output_json is not None
    assert json.loads(output.output_json) == {"text": "raw:first"}
    assert observed == ["first"]

    adapter.unregister_session(prefix_hash)
    assert adapter.register_session(other) == prefix_hash
    replacement_output = await adapter.execute_query_activity(
        adapter.query_activity_input(other, query_id="other", tail="Question", output_name="IdentityAnswer")
    )
    assert replacement_output.output_json is not None
    assert json.loads(replacement_output.output_json) == {"text": "raw:other"}
    assert observed == ["first", "other"]
    adapter.unregister_session(prefix_hash)


@pytest.mark.parametrize("with_validator", [False, True])
def test_matching_validator_identity_allows_session_replacement(with_validator: bool) -> None:
    validator = make_validator("closure", "shared", []) if with_validator else None
    model = Scripted(respond).model()
    first = make_session(model, validator)
    replacement = make_session(model, validator)
    assert first.output_validator is replacement.output_validator
    assert adapter.execution_fingerprint(first) == adapter.execution_fingerprint(replacement)
    prefix_hash = adapter.register_session(first)
    assert adapter.register_session(replacement) == prefix_hash
    assert adapter.register_session(replacement) == prefix_hash
    assert adapter._SESSIONS[prefix_hash] is replacement
    adapter.unregister_session(prefix_hash)
