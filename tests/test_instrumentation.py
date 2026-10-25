"""Actual in-memory OpenTelemetry spans, never a network exporter."""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.instrumented import InstrumentationSettings
from temporalio.testing import ActivityEnvironment
from tests.support import Scripted, text_result, tool_result

from forkreduce import OutputSpec, Policy, Query, build
from forkreduce.contrib import temporal as adapter


class ObservedAnswer(BaseModel):
    """A synthetic integer result."""

    value: int


@pytest.fixture
def tracing():  # type: ignore[no-untyped-def]
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(tracer_provider=provider, include_content=False)
    yield settings, exporter
    provider.shutdown()


def positive(output: Any) -> Any:
    value = output["value"] if isinstance(output, dict) else output.value
    if value < 1:
        raise ModelRetry("value must be positive")
    return output


@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("feedback", [False, True])
async def test_instrumented_requests_include_feedback_and_redact_content(
    tracing, mode: str, dynamic: bool, feedback: bool
) -> None:  # type: ignore[no-untyped-def]
    settings, exporter = tracing

    def reply(_messages, info, number):  # type: ignore[no-untyped-def]
        payload = {"value": -1 if feedback and number == 1 else 7}
        return tool_result(info, info.output_tools[-1].name, payload) if mode == "tools" else text_result(payload)

    scripted = Scripted(reply)
    options = {
        "documents": "synthetic-private-document-marker",
        "instructions": "Answer.",
        "outputs": [] if dynamic else [ObservedAnswer],
        "output_mode": mode,
        "dynamic_outputs": "generic_tool" if dynamic and mode == "tools" else "prompted" if dynamic else None,
        "output_validator": positive,
        "policy": Policy(retries=0, output_feedback_retries=int(feedback)),
        "warmup": False,
    }
    session = build(scripted.model(), **options, instrument=settings)
    baseline = build(scripted.model(), **options)
    assert session.prefix_hash == baseline.prefix_hash
    assert adapter.execution_fingerprint(session) == adapter.execution_fingerprint(baseline)
    output = (
        OutputSpec(
            "Integer",
            {"type": "object", "properties": {"value": {"type": "integer", "minimum": 1}}, "required": ["value"]},
        )
        if dynamic
        else ObservedAnswer
    )
    result = await session.ask(Query(tail="synthetic-private-question-marker", output=output))
    assert result.output == ({"value": 7} if dynamic else ObservedAnswer(value=7))
    chats = [span for span in exporter.get_finished_spans() if span.attributes.get("gen_ai.operation.name") == "chat"]
    invocations = [
        span for span in exporter.get_finished_spans() if span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    ]
    assert len(chats) == result.usage.requests == 1 + int(feedback)
    assert len(invocations) == 1
    assert all(span.attributes.get("gen_ai.agent.name") == "forkreduce" for span in chats)
    encoded = json.dumps([dict(span.attributes) for span in exporter.get_finished_spans()], default=str)
    assert "synthetic-private-document-marker" not in encoded
    assert "synthetic-private-question-marker" not in encoded


async def test_default_instrumentation_remains_disabled_despite_global_default(tracing) -> None:  # type: ignore[no-untyped-def]
    settings, exporter = tracing
    previous = Agent._instrument_default
    Agent.instrument_all(settings)
    try:
        scripted = Scripted(lambda _messages, info, _n: tool_result(info, "ObservedAnswer", {"value": 7}))
        session = build(scripted.model(), documents="Synthetic", instructions="Answer.", outputs=[ObservedAnswer])
        result = await session.ask(Query(tail="Question", output=ObservedAnswer))
        assert result.output == ObservedAnswer(value=7)
        assert exporter.get_finished_spans() == ()
    finally:
        Agent.instrument_all(previous)


@pytest.mark.parametrize("dynamic", [False, True])
async def test_worker_factory_restores_instrumentation_locally(
    tracing, monkeypatch: pytest.MonkeyPatch, dynamic: bool
) -> None:  # type: ignore[no-untyped-def]
    settings, exporter = tracing
    for name in ("_SESSIONS", "_FINGERPRINTS", "_FACTORIES"):
        monkeypatch.setattr(adapter, name, {})
    for name in ("_DEFAULT_FACTORY", "_STORE", "_PREFIX_STORE"):
        monkeypatch.setattr(adapter, name, None)
    scripted = Scripted(lambda _messages, info, _n: tool_result(info, info.output_tools[-1].name, {"value": 7}))
    options = {"outputs": [] if dynamic else [ObservedAnswer], "dynamic_outputs": "generic_tool" if dynamic else None}
    original = build(scripted.model(), documents="Synthetic factory evidence.", instructions="Answer.", **options)
    store = adapter.InMemoryMessageStore()
    prefix_ref = await adapter.freeze_prefix(original, store)
    adapter.set_prefix_store(store)

    def restore(snapshot: adapter.FrozenPrefix):  # type: ignore[no-untyped-def]
        return build(
            scripted.model(),
            documents=snapshot.documents,
            instructions=snapshot.instructions,
            instrument=settings,
            **options,
        )

    adapter.set_default_session_factory(restore)
    spec = (
        OutputSpec("Integer", {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]})
        if dynamic
        else None
    )
    payload = adapter.query_activity_input(
        original,
        query_id="q",
        tail="Question",
        prefix_ref=prefix_ref,
        output_name=None if dynamic else "ObservedAnswer",
        output_spec=spec,
    )
    result = await ActivityEnvironment().run(adapter.execute_query_activity, payload)
    assert json.loads(result.output_json) == {"value": 7}
    assert result.prefix_hash == original.prefix_hash
    chats = [span for span in exporter.get_finished_spans() if span.attributes.get("gen_ai.operation.name") == "chat"]
    assert len(chats) == 1 and chats[0].attributes["gen_ai.agent.name"] == "forkreduce"
    assert "tracer_provider" not in (await store.get(prefix_ref)).decode()
