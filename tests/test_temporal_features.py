"""Serializable Temporal options, reference payloads, heartbeats, and batching."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from temporalio.converter import DataConverter
from temporalio.exceptions import ApplicationError
from tests.support import Scripted, tool_result, user_text

from forkreduce import ActivityOptions, build
from forkreduce.contrib import temporal as adapter
from forkreduce.policy import Policy


class FeatureAnswer(BaseModel):
    """An offline structured answer."""

    text: str


@pytest.fixture(autouse=True)
def isolated_registrations(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(adapter, "_SESSIONS", {})
    monkeypatch.setattr(adapter, "_FINGERPRINTS", {})
    monkeypatch.setattr(adapter, "_FACTORIES", {})
    monkeypatch.setattr(adapter, "_DEFAULT_FACTORY", None)
    monkeypatch.setattr(adapter, "_STORE", None)
    monkeypatch.setattr(adapter, "_PREFIX_STORE", None)


def make_session(*, documents="evidence", activity_options=None, handler=None):  # type: ignore[no-untyped-def]
    if handler is None:

        def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
            return tool_result(info, "FeatureAnswer", {"text": "answer"})

    return build(
        Scripted(handler).model(),
        documents=documents,
        instructions="Use the evidence.",
        outputs=[FeatureAnswer],
        warmup=False,
        policy=Policy(retries=3, retry_backoff=0.25, max_backoff=2),
        activity_options=activity_options,
    )


def payload_for(session, **kwargs):  # type: ignore[no-untyped-def]
    return adapter.query_activity_input(session, query_id="q", tail="question", output_name="FeatureAnswer", **kwargs)


def test_options_copy_and_legacy_retry_inheritance() -> None:
    options = ActivityOptions(start_to_close=180, heartbeat_timeout=30, maximum_attempts=2)
    session = make_session(activity_options=options)
    payload = payload_for(session)
    retry = adapter.activity_retry_policy(payload)
    assert retry.maximum_attempts == 2
    assert retry.initial_interval == timedelta(seconds=0.25)
    assert retry.maximum_interval == timedelta(seconds=2)
    assert adapter.activity_timeout_options(options) == {
        "start_to_close_timeout": timedelta(seconds=180),
        "heartbeat_timeout": timedelta(seconds=30),
    }
    overridden = replace(
        payload,
        activity_options=ActivityOptions(
            retry_backoff=0.5, max_backoff=4, backoff_coefficient=3, schedule_to_close=500
        ),
    )
    retry = adapter.activity_retry_policy(overridden)
    assert retry.maximum_attempts == 4
    assert retry.initial_interval == timedelta(seconds=0.5)
    assert retry.maximum_interval == timedelta(seconds=4)
    assert retry.backoff_coefficient == 3
    assert adapter.activity_timeout_options(overridden.activity_options)["schedule_to_close_timeout"] == timedelta(
        seconds=500
    )
    assert adapter.activity_timeout_options(None) == {"start_to_close_timeout": timedelta(seconds=300)}


@pytest.mark.asyncio
async def test_payload_dataclasses_roundtrip_default_converter() -> None:
    session = make_session(activity_options=ActivityOptions(heartbeat_timeout=10))
    query = payload_for(session, prefix_ref="prefix:ref", history_ref="history:ref")
    config = adapter.MapQueriesInput(
        queries=[query],
        batch_size=2,
        child_concurrency=3,
        activity_options=ActivityOptions(start_to_close=180, maximum_attempts=2),
    )
    hint = list[adapter.QueryActivityInput] | adapter.MapQueriesInput
    for original in ([query], config):
        encoded = await DataConverter.default.encode([original])
        decoded = await DataConverter.default.decode(encoded, [hint])
        assert decoded == [original]
    assert len(json.dumps(asdict(query))) < 1024


@pytest.mark.asyncio
async def test_prefix_factory_reconstruction_and_reference_continuation() -> None:
    observed = []

    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        observed.append(user_text(messages))
        return tool_result(info, "FeatureAnswer", {"text": "answer"})

    session = make_session(documents="large evidence " * 10_000, handler=handler)
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store)
    frozen = adapter.FrozenPrefix(**json.loads(await store.get(ref)))
    assert frozen.prefix_hash == session.prefix_hash
    assert frozen.instructions == session.instructions
    assert frozen.documents == session.documents.documents
    assert "model" not in asdict(frozen)
    rebuilt = []

    async def factory(snapshot):  # type: ignore[no-untyped-def]
        rebuilt.append(snapshot.prefix_hash)
        return build(
            session.model,
            documents=snapshot.documents,
            instructions=snapshot.instructions,
            outputs=[FeatureAnswer],
            warmup=False,
        )

    adapter.register_session_factory(session.prefix_hash, factory)
    adapter.set_prefix_store(store)
    adapter.set_message_store(store)
    query = replace(payload_for(session, prefix_ref=ref), require_references=True)
    result = await adapter.execute_query_activity(query)
    assert result.messages_ref and result.output_ref
    assert result.messages_json is None and result.output_json is None
    assert json.loads(await store.get(result.output_ref)) == {"text": "answer"}
    assert len(json.dumps(asdict(query))) < 1024
    assert len(json.dumps(asdict(result))) < 1024
    await adapter.execute_query_activity(
        replace(query, query_id="q2", tail="continue", history_ref=result.messages_ref, round=1)
    )
    assert rebuilt == [session.prefix_hash, session.prefix_hash]
    assert "question" in observed[1] and "continue" in observed[1]
    assert "large evidence" in observed[1]
    adapter.unregister_session(session.prefix_hash)
    assert session.prefix_hash not in adapter._FACTORIES


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["store", "missing-ref", "factory", "hash", "corrupt", "rebuild"])
async def test_prefix_configuration_and_integrity_error_classification(problem: str) -> None:
    session = make_session()
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store)
    adapter.set_prefix_store(store)
    adapter.register_session_factory(session.prefix_hash, lambda _frozen: session)
    query = payload_for(session, prefix_ref=ref)
    if problem == "store":
        adapter.set_prefix_store(None)
    elif problem == "missing-ref":
        query.prefix_ref = "does-not-exist"
    elif problem == "factory":
        adapter.unregister_session(session.prefix_hash)
    elif problem == "hash":
        query.prefix_hash = "different"
    elif problem == "corrupt":
        snapshot = json.loads(await store.get(ref))
        snapshot["instructions"] = "corrupt rules"
        query.prefix_ref = await store.put("corrupt", json.dumps(snapshot).encode())
    else:
        adapter.register_session_factory(session.prefix_hash, lambda _frozen: make_session(documents="different"))
    with pytest.raises(ApplicationError) as caught:
        await adapter.execute_query_activity(query)
    assert caught.value.non_retryable is (problem != "factory")
    assert caught.value.type == ("session_factory_missing" if problem == "factory" else "permanent")


@pytest.mark.asyncio
async def test_large_output_references_are_explicit_and_defaults_remain_compatible() -> None:
    large = "x" * (64 * 1024)

    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        return tool_result(info, "FeatureAnswer", {"text": large})

    session = make_session(handler=handler)
    adapter.register_session(session)
    store = adapter.InMemoryMessageStore()
    adapter.set_message_store(store)
    result = await adapter.execute_query_activity(payload_for(session))
    assert result.output_json and json.loads(result.output_json) == {"text": large}
    assert result.output_ref is None
    result = await adapter.execute_query_activity(replace(payload_for(session), require_references=True))
    assert result.output_json is None and result.output_ref
    assert json.loads(await store.get(result.output_ref)) == {"text": large}
    adapter.unregister_session(session.prefix_hash)
    small = make_session()
    adapter.register_session(small)
    result = await adapter.execute_query_activity(payload_for(small))
    assert result.output_json and json.loads(result.output_json) == {"text": "answer"}
    assert result.output_ref is None


@pytest.mark.asyncio
async def test_batch_reference_requirements_fail_before_model_call() -> None:
    session = make_session()
    adapter.register_session(session)
    query = replace(payload_for(session), require_references=True)
    with pytest.raises(ApplicationError, match="message store") as caught:
        await adapter.execute_query_activity(query)
    assert caught.value.non_retryable
    adapter.set_message_store(adapter.InMemoryMessageStore())
    with pytest.raises(ApplicationError, match="history_ref"):
        await adapter.execute_query_activity(replace(query, history_json="[]"))


@pytest.mark.asyncio
async def test_heartbeat_helper_is_safe_outside_activity(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(adapter.activity, "in_activity", lambda: False)
    monkeypatch.setattr(adapter.activity, "heartbeat", lambda: pytest.fail("unexpected heartbeat"))
    assert await adapter.heartbeat_while(asyncio.sleep(0, result="done"), 0.01) == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_periodic_heartbeat_task_always_stops(monkeypatch, outcome: str):  # type: ignore[no-untyped-def]
    beats = []
    monkeypatch.setattr(adapter.activity, "in_activity", lambda: True)
    monkeypatch.setattr(adapter.activity, "heartbeat", lambda: beats.append(True))
    started = asyncio.Event()

    async def work():  # type: ignore[no-untyped-def]
        started.set()
        await asyncio.sleep(0.04)
        if outcome == "failure":
            raise RuntimeError("model failure")
        return "done"

    existing = asyncio.all_tasks()
    running = asyncio.create_task(adapter.heartbeat_while(work(), 0.015))
    await started.wait()
    if outcome == "cancel":
        await asyncio.sleep(0.02)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    elif outcome == "failure":
        with pytest.raises(RuntimeError, match="model failure"):
            await running
    else:
        assert await running == "done"
    assert len(beats) >= 2
    count = len(beats)
    await asyncio.sleep(0.02)
    assert len(beats) == count
    assert asyncio.all_tasks() == existing


def output_for(query):  # type: ignore[no-untyped-def]
    return adapter.QueryActivityOutput(
        query_id=query.query_id,
        prefix_hash=query.prefix_hash,
        output_json=None,
        output_type="FeatureAnswer",
        cache_hit_tokens=0,
        attempts=1,
        messages_ref=f"messages:{query.query_id}",
        output_ref=f"output:{query.query_id}",
    )


def test_child_ids_are_stable_and_scoped_to_parent_execution(monkeypatch):  # type: ignore[no-untyped-def]
    info = SimpleNamespace(workflow_id="parent-a", run_id="run-a")
    monkeypatch.setattr(adapter.workflow, "info", lambda: info)
    first = adapter.child_workflow_id("prefix", 0)
    assert adapter.child_workflow_id("prefix", 0) == first
    assert adapter.child_workflow_id("prefix", 1) != first
    assert adapter.child_workflow_id("other-prefix", 0) != first
    info.run_id = "run-b"
    assert adapter.child_workflow_id("prefix", 0) == first
    info.run_id = "run-a"
    info.workflow_id = "parent-b"
    assert adapter.child_workflow_id("prefix", 0) != first


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [{"batch_size": 0}, {"batch_size": -1}, {"child_concurrency": 0}])
async def test_invalid_batch_configuration_is_permanent(changes):  # type: ignore[no-untyped-def]
    config = replace(adapter.MapQueriesInput(queries=[], batch_size=2), **changes)
    with pytest.raises(ApplicationError) as caught:
        await adapter.MapQueriesWorkflow().run(config)
    assert caught.value.non_retryable


@pytest.mark.asyncio
async def test_batch_rejects_inline_history_and_activity_results(monkeypatch):  # type: ignore[no-untyped-def]
    query = payload_for(make_session(), history_json="[]")
    with pytest.raises(ApplicationError, match="history_ref"):
        await adapter.MapQueriesWorkflow().run(adapter.MapQueriesInput(queries=[query], batch_size=2))
    with pytest.raises(ApplicationError, match="history_ref"):
        await adapter.QueryBatchWorkflow().run(adapter.MapQueriesInput(queries=[query]))

    async def execute_activity(_fn, payload, **kwargs):  # type: ignore[no-untyped-def]
        return replace(output_for(payload), messages_json="[]", messages_ref=None)

    monkeypatch.setattr(adapter.workflow, "execute_activity", execute_activity)
    with pytest.raises(ApplicationError, match="references") as caught:
        await adapter.QueryBatchWorkflow().run(adapter.MapQueriesInput(queries=[replace(query, history_json=None)]))
    assert caught.value.non_retryable


@pytest.mark.parametrize("forced", [False, True])
async def test_auto_prefix_restores_resolved_mode_and_result_metadata(forced: bool) -> None:
    model = TestModel(
        custom_output_args={"text": "restored"} if forced else None,
        custom_output_text=None if forced else '{"text":"restored"}',
        profile={"supports_forced_tool_choice": forced},
    )
    session = build(
        model,
        documents="frozen auto document",
        instructions="Use the evidence.",
        outputs=[FeatureAnswer],
        on_unforced_tools="error",
    )
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store)
    snapshot = adapter.FrozenPrefix(**json.loads(await store.get(ref)))
    expected = "tools" if forced else "prompted"
    assert snapshot.output_mode == "auto" and snapshot.resolved_output_mode == expected
    snapshot.validate()

    def restore(frozen):  # type: ignore[no-untyped-def]
        return build(
            model,
            documents=frozen.documents,
            instructions=frozen.instructions,
            outputs=[FeatureAnswer],
            output_mode=frozen.output_mode,
            on_unforced_tools=frozen.on_unforced_tools,
        )

    adapter.register_session_factory(session.prefix_hash, restore)
    adapter.set_prefix_store(store)
    result = await adapter.execute_query_activity(payload_for(session, prefix_ref=ref))
    assert json.loads(result.output_json) == {"text": "restored"}
    assert result.resolved_output_mode == expected
    fingerprint = json.loads(adapter.execution_fingerprint(restore(snapshot)))
    assert fingerprint["resolved_output_mode"] == expected
    assert fingerprint["on_unforced_tools"] == "error"


async def test_legacy_snapshot_without_resolved_mode_is_still_readable() -> None:
    session = build(
        TestModel(custom_output_args={"text": "legacy"}),
        documents="old snapshot",
        instructions="Use the evidence.",
        outputs=[FeatureAnswer],
        output_mode="tools",
    )
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store)
    record = json.loads(await store.get(ref))
    del record["resolved_output_mode"]
    del record["on_unforced_tools"]
    old_ref = await store.put("legacy", json.dumps(record).encode())
    adapter.FrozenPrefix(**record).validate()
    adapter.set_prefix_store(store)
    adapter.register_session_factory(session.prefix_hash, lambda _snapshot: session)
    result = await adapter.execute_query_activity(payload_for(session, prefix_ref=old_ref))
    assert json.loads(result.output_json) == {"text": "legacy"}
    assert result.resolved_output_mode == "tools"


async def test_worker_rejects_changed_auto_resolution_during_restore() -> None:
    original = make_session()
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(original, store)
    changed = TestModel(profile={"supports_forced_tool_choice": False})
    adapter.set_prefix_store(store)
    adapter.register_session_factory(
        original.prefix_hash,
        lambda frozen: build(
            changed,
            documents=frozen.documents,
            instructions=frozen.instructions,
            outputs=[FeatureAnswer],
            output_mode=frozen.output_mode,
        ),
    )
    with pytest.raises(ApplicationError, match="prefix hash") as caught:
        await adapter.execute_query_activity(payload_for(original, prefix_ref=ref))
    assert caught.value.non_retryable


async def test_temporal_thinking_mode_flip_is_a_permanent_caller_error() -> None:
    scripted = Scripted(lambda _messages, info, _n: tool_result(info, "FeatureAnswer", {"text": "unused"}))
    model = FunctionModel(
        scripted.model().function,
        profile={
            "supports_thinking": True,
            "supports_forced_tool_choice_with_thinking": False,
        },
    )
    session = build(model, documents="mode conflict", instructions="Use evidence.", outputs=[FeatureAnswer])
    adapter.register_session(session)
    payload = payload_for(session, model_settings={"thinking": True})
    with pytest.raises(ApplicationError) as caught:
        await adapter.execute_query_activity(payload)
    assert caught.value.non_retryable
    assert not scripted.calls
