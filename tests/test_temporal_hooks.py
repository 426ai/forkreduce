"""Composable query primitives and immediate per-query hooks on offline workers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError, RetryState
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from pydantic_ai.exceptions import ModelHTTPError
    from temporalio.client import WorkflowFailureError
    from tests.support import Scripted, tool_result, user_text
    from tests.test_temporal_runtime import sandbox_runner, server_options

    from forkreduce import ActivityOptions, build
    from forkreduce.contrib import temporal as adapter
    from forkreduce.policy import Policy


class HookAnswer(BaseModel):
    text: str


@pytest.fixture(autouse=True)
def isolated_registrations(monkeypatch):  # type: ignore[no-untyped-def]
    for name in ("_SESSIONS", "_FINGERPRINTS", "_FACTORIES"):
        monkeypatch.setattr(adapter, name, {})
    for name in ("_DEFAULT_FACTORY", "_STORE", "_PREFIX_STORE"):
        monkeypatch.setattr(adapter, name, None)


@workflow.defn(name="synthetic.custom_batch")
class CustomBatchWorkflow:
    @workflow.run
    async def run(self, config: adapter.MapQueriesInput) -> list[adapter.QueryActivityOutput]:
        return await adapter.QueryBatchWorkflow().run(config)


@workflow.defn
class ComposedQueriesWorkflow:
    @workflow.run
    async def run(self, config: adapter.MapQueriesInput) -> list[adapter.QueryActivityOutput]:
        results = []
        for batch in adapter.plan_batches(config.queries, 1):
            for query in batch:
                query = adapter.require_query_references(query)
                try:
                    output = await adapter.execute_query(
                        query,
                        ActivityOptions(maximum_attempts=1),
                        task_queue=config.task_queue,
                        require_references=True,
                        activity_name="synthetic.composed_query",
                    )
                except ActivityError as error:
                    output = adapter.query_failure(query, error)
                adapter.validate_query_references(output)
                results.append(output)
        return results


@asynccontextmanager
async def running_map(config, hook, *, plugin, run_query=None, extra_workflows=()):  # type: ignore[no-untyped-def]
    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    workflows = [adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow, *extra_workflows]
    async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
        queue = f"query-hooks-{uuid4()}"
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=workflows,
            activities=[run_query or adapter.run_query_activity, hook],
            workflow_runner=runner,
            max_heartbeat_throttle_interval=timedelta(milliseconds=20),
            default_heartbeat_throttle_interval=timedelta(milliseconds=20),
        ):
            handle = await env.client.start_workflow(
                adapter.MapQueriesWorkflow.run,
                replace(config, task_queue=queue),
                id=f"query-hooks-{uuid4()}",
                task_queue=queue,
                execution_timeout=timedelta(seconds=30),
            )
            yield env, handle, queue
            history = await handle.fetch_history()
            children = [
                event.start_child_workflow_execution_initiated_event_attributes.workflow_id
                for event in history.events
                if event.HasField("start_child_workflow_execution_initiated_event_attributes")
            ]
            child_histories = [await env.client.get_workflow_handle(child).fetch_history() for child in children]
        replayer = Replayer(workflows=workflows, plugins=plugins, workflow_runner=runner)
        await replayer.replay_workflow(history)
        for child_history in child_histories:
            await replayer.replay_workflow(child_history)


def make_query(handler=None, *, query_id="root", max_attempts=4):  # type: ignore[no-untyped-def]
    if handler is None:

        def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
            return tool_result(info, "HookAnswer", {"text": "answer"})

    session = build(
        Scripted(handler).model(),
        documents="Synthetic hook evidence.",
        instructions="Use the evidence.",
        outputs=[HookAnswer],
        warmup=False,
        policy=Policy(retries=max_attempts - 1, retry_backoff=0.001, max_backoff=0.001),
    )
    adapter.register_session(session)
    adapter.set_message_store(adapter.InMemoryMessageStore())
    return adapter.query_activity_input(
        session, query_id=query_id, tail="initial question", output_name="HookAnswer", meta={"synthetic": True}
    )


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("mode", ["direct", "child", "custom"])
async def test_repair_round_ids_history_metadata_custom_batch_and_replay(plugin: bool, mode: str) -> None:
    requests = []
    hooks = []
    activity_ids = []

    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        requests.append(user_text(messages))
        activity_ids.append(activity.info().activity_id)
        return tool_result(info, "HookAnswer", {"text": "answer"})

    query = make_query(handler)
    shared_meta = {"nested": {"values": ["first"]}, "removed": True}

    @activity.defn(name="synthetic.repair")
    async def repair(payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        hooks.append((payload, activity.info()))
        assert payload.output.messages_ref and payload.output.output_ref
        assert payload.output.messages_json is None and payload.output.output_json is None
        assert payload.meta == {"synthetic": True}
        if payload.round == 0:
            return adapter.AfterQueryDecision("follow_up", follow_up_tail="repair one", result_meta=shared_meta)
        if payload.round == 1:
            # A complete replacement query inherits the previous messages reference,
            # but its proposed id and round are never authoritative.
            replacement = replace(payload.query, query_id="ignored", tail="repair two", history_ref=None, round=90)
            return adapter.AfterQueryDecision(
                "follow_up", follow_up_query=replacement, result_meta={"final": [2, True, None]}
            )
        return adapter.AfterQueryDecision("done")

    config = adapter.MapQueriesInput(
        queries=[query],
        batch_size=None if mode == "direct" else 1,
        batch_workflow="synthetic.custom_batch" if mode == "custom" else None,
        activity_concurrency=1,
        warmup="per_batch",
        after_query_activity="synthetic.repair",
        max_rounds=2,
    )
    extras = (CustomBatchWorkflow,) if mode == "custom" else ()
    async with running_map(config, repair, plugin=plugin, extra_workflows=extras) as (_env, handle, queue):
        result = (await asyncio.wait_for(handle.result(), 30))[0]
        assert result.query_id == "root#r2" and result.failure_kind is None
        assert result.result_meta == {"final": [2, True, None]}
        assert [payload.query.query_id for payload, _info in hooks] == ["root", "root#r1", "root#r2"]
        assert [payload.round for payload, _info in hooks] == [0, 1, 2]
        assert hooks[1][0].query.history_ref == hooks[0][0].output.messages_ref
        assert hooks[2][0].query.history_ref == hooks[1][0].output.messages_ref
        assert hooks[1][0].output.result_meta == shared_meta
        assert hooks[2][0].output.result_meta == {"final": [2, True, None]}
        assert all(payload.parent_workflow_id == handle.id for payload, _info in hooks)
        assert all(payload.batch_index == 0 for payload, _info in hooks)
        assert all(info.task_queue == queue for _payload, info in hooks)
        assert [info.activity_id for _payload, info in hooks] == [
            f"{adapter.activity_id(query.prefix_hash, query_id)}:after_query:r{round_index}"
            for round_index, query_id in enumerate(("root", "root#r1", "root#r2"))
        ]
        assert activity_ids == [
            adapter.activity_id(query.prefix_hash, query_id) for query_id in ("root", "root#r1", "root#r2")
        ]
        assert "initial question" in requests[2] and "repair one" in requests[2] and "repair two" in requests[2]
        assert len({payload.output.messages_ref for payload, _info in hooks}) == 3
        history = await handle.fetch_history()
        children = [
            event.start_child_workflow_execution_initiated_event_attributes.workflow_id
            for event in history.events
            if event.HasField("start_child_workflow_execution_initiated_event_attributes")
        ]
        assert children == ([] if mode == "direct" else [adapter.batch_child_id(handle.id, query.prefix_hash, 0)])


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("decision", ["cap", "failed", "prefix", "oversize"])
async def test_hook_decision_failures_are_safe_and_round_cap_is_inclusive(plugin: bool, decision: str) -> None:
    query = make_query()
    seen = []

    @activity.defn(name="synthetic.reject")
    async def reject(payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        seen.append(payload.round)
        if decision == "cap":
            return adapter.AfterQueryDecision("follow_up", follow_up_tail="another repair")
        if decision == "prefix":
            return adapter.AfterQueryDecision(
                "follow_up", follow_up_query=replace(payload.query, prefix_hash="different-prefix")
            )
        if decision == "oversize":
            return adapter.AfterQueryDecision("done", result_meta={"body": "x" * (16 * 1024)})
        return adapter.AfterQueryDecision("failed")

    config = adapter.MapQueriesInput(queries=[query], after_query_activity="synthetic.reject")
    async with running_map(config, reject, plugin=plugin) as (_env, handle, _queue):
        result = (await asyncio.wait_for(handle.result(), 30))[0]
        assert seen == ([0, 1] if decision == "cap" else [0])
        assert result.query_id == ("root#r1" if decision == "cap" else "root")
        assert result.failure_kind == "permanent" and result.failure_type == "after_query_failed"
        assert result.output_ref is None and result.messages_ref is None
        assert result.output_json is None and result.messages_json is None
        assert "different-prefix" not in json.dumps(asdict(result))


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("mode", ["raise", "return"])
@pytest.mark.parametrize("retry_hook", [False, True])
async def test_hook_activity_errors_have_independent_finite_retry_policy(
    plugin: bool, mode: str, retry_hook: bool
) -> None:
    query = make_query(max_attempts=4)
    attempts = []

    @activity.defn(name="synthetic.hook_error")
    async def broken(_payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        attempts.append(activity.info().attempt)
        raise ApplicationError("synthetic-private-hook-message", type="budget", non_retryable=False)

    options = (
        ActivityOptions(start_to_close=7, maximum_attempts=2, retry_backoff=0.001, max_backoff=0.001)
        if retry_hook
        else None
    )
    config = adapter.MapQueriesInput(
        queries=[query], after_query_activity="synthetic.hook_error", after_query_options=options, on_query_failure=mode
    )
    async with running_map(config, broken, plugin=plugin) as (_env, handle, _queue):
        if mode == "raise":
            with pytest.raises(WorkflowFailureError):
                await asyncio.wait_for(handle.result(), 30)
        else:
            result = (await asyncio.wait_for(handle.result(), 30))[0]
            assert result.failure_kind == "unknown" and result.failure_type == "after_query_failed"
            assert "synthetic-private-hook-message" not in json.dumps(asdict(result))
            assert result.messages_ref is None and result.output_ref is None
        assert attempts == ([1, 2] if retry_hook else [1])


@pytest.mark.parametrize("plugin", [False, True])
async def test_return_model_failure_still_runs_hook_and_can_repair(plugin: bool) -> None:
    seen = []

    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        if activity.info().activity_id.endswith(":root"):
            raise ModelHTTPError(401, "synthetic-private-model-error")
        return tool_result(info, "HookAnswer", {"text": "repaired"})

    query = make_query(handler)

    @activity.defn(name="synthetic.repair_failure")
    async def repair(payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        seen.append(payload)
        if payload.round == 0:
            assert payload.output.failure_kind == "permanent"
            assert payload.output.messages_ref is None
            return adapter.AfterQueryDecision("follow_up", follow_up_tail="repair failed attempt")
        return adapter.AfterQueryDecision("done", result_meta={"repaired": True})

    config = adapter.MapQueriesInput(
        queries=[query], after_query_activity="synthetic.repair_failure", on_query_failure="return"
    )
    async with running_map(config, repair, plugin=plugin) as (_env, handle, _queue):
        result = (await asyncio.wait_for(handle.result(), 30))[0]
        assert result.failure_kind is None and result.query_id == "root#r1"
        assert result.result_meta == {"repaired": True}
        assert len(seen) == 2


@pytest.mark.parametrize("plugin", [False, True])
async def test_hook_runs_before_unrelated_slow_query_finishes(plugin: bool) -> None:
    query = make_query()
    fast_hooked = asyncio.Event()
    trace = []

    @activity.defn(name="forkreduce.run_query")
    async def gated(payload: adapter.QueryActivityInput) -> adapter.QueryActivityOutput:
        trace.append(("query_start", payload.query_id))
        if payload.query_id == "slow":
            await asyncio.wait_for(fast_hooked.wait(), 10)
        output = await adapter.run_query_activity(payload)
        trace.append(("query_finish", payload.query_id))
        return output

    @activity.defn(name="synthetic.immediate")
    async def immediate(payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        trace.append(("hook", payload.query.query_id))
        if payload.query.query_id == "fast":
            fast_hooked.set()
        return adapter.AfterQueryDecision("done")

    config = adapter.MapQueriesInput(
        queries=[replace(query, query_id="slow"), replace(query, query_id="fast")],
        batch_size=2,
        activity_concurrency=2,
        after_query_activity="synthetic.immediate",
    )
    async with running_map(config, immediate, plugin=plugin, run_query=gated) as (_env, handle, _queue):
        results = await asyncio.wait_for(handle.result(), 30)
        assert [result.query_id for result in results] == ["slow", "fast"]
        assert trace.index(("hook", "fast")) < trace.index(("query_finish", "slow"))


@pytest.mark.parametrize("plugin", [False, True])
async def test_concurrency_slot_covers_hook_and_all_repairs(plugin: bool) -> None:
    query = make_query()
    trace = []

    @activity.defn(name="forkreduce.run_query")
    async def recorded(payload: adapter.QueryActivityInput) -> adapter.QueryActivityOutput:
        trace.append(("query", payload.query_id))
        return await adapter.run_query_activity(payload)

    @activity.defn(name="synthetic.serial")
    async def serial(payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        trace.append(("hook", payload.query.query_id))
        await asyncio.sleep(0.01)
        if payload.round == 0:
            return adapter.AfterQueryDecision("follow_up", follow_up_tail="repair")
        return adapter.AfterQueryDecision("done")

    config = adapter.MapQueriesInput(
        queries=[replace(query, query_id=name) for name in ("a", "b")],
        activity_concurrency=1,
        after_query_activity="synthetic.serial",
    )
    async with running_map(config, serial, plugin=plugin, run_query=recorded) as (_env, handle, _queue):
        await asyncio.wait_for(handle.result(), 30)
        assert trace == [
            (kind, name) for root in ("a", "b") for name in (root, f"{root}#r1") for kind in ("query", "hook")
        ]


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("batch_size", [None, 1])
async def test_hook_workflow_cancellation_propagates(plugin: bool, batch_size: int | None) -> None:
    query = make_query()
    entered = asyncio.Event()
    stopped = asyncio.Event()

    @activity.defn(name="synthetic.cancel_hook")
    async def cancelled(payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        entered.set()
        try:
            await adapter.heartbeat_while(asyncio.Event().wait(), 0.1)
        finally:
            stopped.set()
        return adapter.AfterQueryDecision("done")

    config = adapter.MapQueriesInput(
        queries=[query],
        batch_size=batch_size,
        after_query_activity="synthetic.cancel_hook",
        after_query_options=ActivityOptions(start_to_close=20, heartbeat_timeout=0.1),
        on_query_failure="return",
    )
    async with running_map(config, cancelled, plugin=plugin) as (env, handle, _queue):
        with env.auto_time_skipping_disabled():
            await asyncio.wait_for(entered.wait(), 10)
            await handle.cancel()
            with pytest.raises(WorkflowFailureError) as caught:
                await asyncio.wait_for(handle.result(), 10)
            cause = caught.value.__cause__
            assert cause is not None
            await asyncio.wait_for(stopped.wait(), 10)


@pytest.mark.parametrize("plugin", [False, True])
async def test_public_primitives_compose_in_an_application_workflow(plugin: bool) -> None:
    seen = []

    @activity.defn(name="synthetic.composed_query")
    async def composed(payload: adapter.QueryActivityInput) -> adapter.QueryActivityOutput:
        seen.append(payload)
        if payload.query_id == "bad":
            raise ApplicationError("synthetic-private-error", type="permanent", non_retryable=True)
        return adapter.QueryActivityOutput(
            payload.query_id,
            payload.prefix_hash,
            None,
            "HookAnswer",
            0,
            1,
            messages_ref="synthetic:messages",
            output_ref="synthetic:output",
        )

    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
        queue = f"public-primitives-{uuid4()}"
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[ComposedQueriesWorkflow],
            activities=[composed],
            workflow_runner=runner,
        ):
            handle = await env.client.start_workflow(
                ComposedQueriesWorkflow.run,
                adapter.MapQueriesInput(
                    queries=[
                        adapter.QueryActivityInput("prefix", name, "question", "HookAnswer") for name in ("ok", "bad")
                    ],
                    task_queue=queue,
                ),
                id=f"public-primitives-{uuid4()}",
                task_queue=queue,
            )
            results = await asyncio.wait_for(handle.result(), 30)
            assert [result.failure_kind for result in results] == [None, "permanent"]
            assert all(payload.require_references for payload in seen)
            history = await handle.fetch_history()
        await Replayer(workflows=[ComposedQueriesWorkflow], plugins=plugins, workflow_runner=runner).replay_workflow(
            history
        )


def test_plan_batches_and_child_id_preserve_original_algorithm() -> None:
    queries = [object(), object(), object()]
    assert adapter.plan_batches(queries, 2) == [queries[:2], queries[2:]]
    assert adapter.plan_batches([], 2) == []
    expected = f"forkreduce:{hashlib.sha256(b'parent').hexdigest()}:prefix:batch:3"
    assert adapter.batch_child_id("parent", "prefix", 3) == expected
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ApplicationError):
            adapter.plan_batches(queries, invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
async def test_max_rounds_requires_a_positive_integer(value) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ApplicationError, match="max_rounds"):
        await adapter.MapQueriesWorkflow().run(adapter.MapQueriesInput(queries=[], max_rounds=value))


async def test_reference_enforcement_and_result_metadata_are_copied(monkeypatch):  # type: ignore[no-untyped-def]
    query = adapter.QueryActivityInput("prefix", "root", "question", "HookAnswer", max_attempts=99)
    required = adapter.require_query_references(query)
    assert required is not query and required.require_references and not query.require_references
    with pytest.raises(ApplicationError, match="history_ref"):
        adapter.require_query_references(replace(query, history_json="[]"))
    shared = {"nested": {"items": [1]}}

    async def execute(name, payload, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(payload, adapter.AfterQueryInput):
            if payload.round == 0:
                return adapter.AfterQueryDecision("follow_up", follow_up_tail="repair", result_meta=shared)
            shared["nested"]["items"].append(2)
            assert payload.output.result_meta == {"nested": {"items": [1]}}
            return adapter.AfterQueryDecision("done")
        return adapter.QueryActivityOutput(
            payload.query_id,
            payload.prefix_hash,
            None,
            "HookAnswer",
            0,
            1,
            messages_ref="messages",
            output_ref="output",
        )

    monkeypatch.setattr(adapter.workflow, "execute_activity", execute)
    monkeypatch.setattr(adapter.workflow, "info", lambda: SimpleNamespace(parent=None, workflow_id="map"))
    results = await adapter.MapQueriesWorkflow().run(
        adapter.MapQueriesInput(
            queries=[query],
            after_query_activity="synthetic.copy",
            after_query_options=ActivityOptions(start_to_close=9),
        )
    )
    assert results[0].result_meta == {"nested": {"items": [1]}}


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("parent_cancelled", [False, True])
async def test_hook_cancellation_distinguishes_activity_from_parent(monkeypatch, wrapped: bool, parent_cancelled: bool):  # type: ignore[no-untyped-def]
    query = adapter.QueryActivityInput("prefix", "root", "question", "HookAnswer")
    cause = CancelledError("synthetic cancellation")
    if wrapped:
        error = ActivityError(
            "wrapper",
            scheduled_event_id=1,
            started_event_id=2,
            identity="offline",
            activity_type="hook",
            activity_id="hook",
            retry_state=RetryState.CANCEL_REQUESTED,
        )
        error.__cause__ = cause
    else:
        error = cause

    async def execute(name, payload, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(payload, adapter.AfterQueryInput):
            raise error
        return adapter.QueryActivityOutput("root", "prefix", None, None, 0, 1, messages_ref="messages")

    monkeypatch.setattr(adapter.workflow, "execute_activity", execute)
    monkeypatch.setattr(adapter.workflow, "info", lambda: SimpleNamespace(parent=None, workflow_id="map"))
    monkeypatch.setattr(adapter.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(adapter.workflow, "cancellation_reason", lambda: "" if parent_cancelled else None)
    config = adapter.MapQueriesInput(
        queries=[query], after_query_activity="synthetic.cancel", on_query_failure="return"
    )
    if parent_cancelled or not wrapped:
        with pytest.raises(type(error)):
            await adapter.MapQueriesWorkflow().run(config)
    else:
        result = (await adapter.MapQueriesWorkflow().run(config))[0]
        assert result.failure_kind == "unknown" and result.failure_type == "after_query_failed"


@pytest.mark.parametrize("plugin", [False, True])
async def test_custom_batch_receives_indexes_and_preserves_configuration(plugin: bool) -> None:
    query = make_query()
    seen = []

    @activity.defn(name="synthetic.indexes")
    async def indexes(payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        seen.append(payload)
        return adapter.AfterQueryDecision("done")

    config = adapter.MapQueriesInput(
        queries=[replace(query, query_id=f"q{index}") for index in range(3)],
        batch_size=1,
        batch_workflow="synthetic.custom_batch",
        child_concurrency=1,
        activity_concurrency=1,
        warmup="per_batch",
        after_query_activity="synthetic.indexes",
    )
    async with running_map(config, indexes, plugin=plugin, extra_workflows=(CustomBatchWorkflow,)) as (
        _env,
        handle,
        _queue,
    ):
        results = await asyncio.wait_for(handle.result(), 30)
        assert [result.query_id for result in results] == ["q0", "q1", "q2"]
        assert [payload.batch_index for payload in seen] == [0, 1, 2]
        assert all(payload.parent_workflow_id == handle.id for payload in seen)


@pytest.mark.parametrize("invalid", [{"value": float("nan")}, {"value": {1, 2}}, {1: "not-string"}])
async def test_invalid_result_metadata_fails_safely(monkeypatch, invalid):  # type: ignore[no-untyped-def]
    query = adapter.QueryActivityInput("prefix", "root", "question", "HookAnswer")

    async def execute(name, payload, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(payload, adapter.AfterQueryInput):
            return adapter.AfterQueryDecision("done", result_meta=invalid)
        return adapter.QueryActivityOutput("root", "prefix", None, None, 0, 1, messages_ref="messages")

    monkeypatch.setattr(adapter.workflow, "execute_activity", execute)
    monkeypatch.setattr(adapter.workflow, "info", lambda: SimpleNamespace(parent=None, workflow_id="map"))
    result = (
        await adapter.MapQueriesWorkflow().run(
            adapter.MapQueriesInput(queries=[query], after_query_activity="synthetic.metadata")
        )
    )[0]
    assert result.failure_kind == "permanent" and result.failure_type == "after_query_failed"
    assert result.result_meta is None


def test_result_metadata_limit_counts_compact_utf8_json() -> None:
    exact = {"k": "x" * (16 * 1024 - 8)}
    assert adapter._copy_result_meta(exact) == exact
    with pytest.raises(ValueError, match="16 KiB"):
        adapter._copy_result_meta({"k": "x" * (16 * 1024 - 7)})
    with pytest.raises(ValueError, match="16 KiB"):
        adapter._copy_result_meta({"k": "界" * (6 * 1024)})


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("hook_error", [False, True])
async def test_hook_failure_retains_observed_output_feedback(plugin: bool, hook_error: bool) -> None:
    def reply(_messages, info, number):  # type: ignore[no-untyped-def]
        return tool_result(info, "HookAnswer", {"wrong": "synthetic"} if number == 1 else {"text": "accepted"})

    scripted = Scripted(reply)
    session = build(
        scripted.model(),
        documents="Synthetic feedback evidence.",
        instructions="Answer.",
        outputs=[HookAnswer],
        policy=Policy(retries=0, output_feedback_retries=1),
        warmup=False,
    )
    adapter.register_session(session)
    adapter.set_message_store(adapter.InMemoryMessageStore())
    query = adapter.query_activity_input(session, query_id="feedback", tail="Question", output_name="HookAnswer")

    @activity.defn(name="synthetic.feedback_review")
    async def reject(_input: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        if hook_error:
            raise ApplicationError("synthetic rejection", non_retryable=True)
        return adapter.AfterQueryDecision("failed")

    config = adapter.MapQueriesInput(
        queries=[query], after_query_activity="synthetic.feedback_review", on_query_failure="return"
    )
    async with running_map(config, reject, plugin=plugin) as (_env, handle, _queue):
        result = (await handle.result())[0]
        assert result.failure_type == "after_query_failed"
        assert result.feedback_retries_used == 1 and result.meta == {"feedback_retries_used": 1}
        assert len(scripted.calls) == 2
