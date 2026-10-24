"""Scheduling limits, queue routing, and warmup through real offline workers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Literal
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.models.test import TestModel
    from temporalio.client import WorkflowFailureError
    from temporalio.exceptions import ApplicationError
    from tests.support import Scripted, tool_result
    from tests.test_temporal_runtime import sandbox_runner, server_options

    from forkreduce import ActivityOptions, build
    from forkreduce.contrib import temporal as adapter
    from forkreduce.policy import Policy


class SchedulingAnswer(BaseModel):
    text: str


@pytest.fixture(autouse=True)
def isolated_registrations(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(adapter, "_SESSIONS", {})
    monkeypatch.setattr(adapter, "_FINGERPRINTS", {})
    monkeypatch.setattr(adapter, "_FACTORIES", {})
    monkeypatch.setattr(adapter, "_DEFAULT_FACTORY", None)
    monkeypatch.setattr(adapter, "_STORE", None)
    monkeypatch.setattr(adapter, "_PREFIX_STORE", None)


@dataclass
class ActivityProbe:
    trace: list[tuple[str, str]] = field(default_factory=list)
    queues: list[str] = field(default_factory=list)
    workflows: dict[str, str] = field(default_factory=dict)
    finished: dict[str, asyncio.Event] = field(default_factory=dict)
    active: int = 0
    peak: int = 0

    def completion(self, query_id: str) -> asyncio.Event:
        return self.finished.setdefault(query_id, asyncio.Event())

    def entrypoint(self):  # type: ignore[no-untyped-def]
        probe = self

        @activity.defn(name="forkreduce.run_query")
        async def recorded(payload: adapter.QueryActivityInput) -> adapter.QueryActivityOutput:
            current = activity.info()
            probe.queues.append(current.task_queue)
            probe.workflows[payload.query_id] = current.workflow_id
            probe.trace.append(("start", payload.query_id))
            probe.active += 1
            probe.peak = max(probe.peak, probe.active)
            try:
                # Include work outside the model request in the measured activity lifetime.
                await asyncio.sleep(0.03)
                return await adapter.run_query_activity(payload)
            finally:
                await asyncio.sleep(0.03)
                probe.active -= 1
                probe.trace.append(("finish", payload.query_id))
                probe.completion(payload.query_id).set()

        return recorded


def make_session(model):  # type: ignore[no-untyped-def]
    session = build(
        model,
        documents="Synthetic scheduling evidence.",
        instructions="Use the registered output.",
        outputs=[SchedulingAnswer],
        warmup=False,
        policy=Policy(retries=0),
        activity_options=ActivityOptions(start_to_close=20, maximum_attempts=1),
    )
    adapter.register_session(session)
    adapter.set_message_store(adapter.InMemoryMessageStore())
    return session


def queries_for(session, count: int = 6):  # type: ignore[no-untyped-def]
    return [
        adapter.query_activity_input(
            session, query_id=f"q{index}", tail=f"Question {index}", output_name="SchedulingAnswer"
        )
        for index in range(count)
    ]


@asynccontextmanager
async def running_map(  # type: ignore[no-untyped-def]
    config, probe: ActivityProbe, *, plugin: bool, separate_queue: bool = False
):
    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
        parent_queue = f"scheduling-parent-{uuid4()}"
        work_queue = f"scheduling-work-{uuid4()}" if separate_queue else parent_queue
        if separate_queue:
            config = replace(config, task_queue=work_queue)
        parent_workflows = [adapter.MapQueriesWorkflow]
        if not separate_queue:
            parent_workflows.append(adapter.QueryBatchWorkflow)
        async with Worker(
            env.client,
            task_queue=parent_queue,
            workflows=parent_workflows,
            activities=[] if separate_queue else [probe.entrypoint()],
            workflow_runner=runner,
        ):
            if separate_queue:
                async with Worker(
                    env.client,
                    task_queue=work_queue,
                    workflows=[adapter.QueryBatchWorkflow],
                    activities=[probe.entrypoint()],
                    workflow_runner=runner,
                ):
                    handle = await env.client.start_workflow(
                        adapter.MapQueriesWorkflow.run,
                        config,
                        id=f"scheduling-{uuid4()}",
                        task_queue=parent_queue,
                        execution_timeout=timedelta(seconds=30),
                    )
                    yield env, handle, parent_queue, work_queue
            else:
                handle = await env.client.start_workflow(
                    adapter.MapQueriesWorkflow.run,
                    config,
                    id=f"scheduling-{uuid4()}",
                    task_queue=parent_queue,
                    execution_timeout=timedelta(seconds=30),
                )
                yield env, handle, parent_queue, work_queue

        history = await handle.fetch_history()
        children = [
            event.start_child_workflow_execution_initiated_event_attributes
            for event in history.events
            if event.HasField("start_child_workflow_execution_initiated_event_attributes")
        ]
        replayer = Replayer(
            workflows=[adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow],
            plugins=plugins,
            workflow_runner=runner,
        )
        await replayer.replay_workflow(history)
        for child in children:
            await replayer.replay_workflow(await env.client.get_workflow_handle(child.workflow_id).fetch_history())


async def assert_outputs(results, session, *, failed_first: bool = False):  # type: ignore[no-untyped-def]
    assert [result.query_id for result in results] == [f"q{index}" for index in range(6)]
    assert all(result.prefix_hash == session.prefix_hash for result in results)
    assert [result.failure_kind for result in results] == (["permanent"] + [None] * 5 if failed_first else [None] * 6)
    for result in results:
        if result.failure_kind is not None:
            assert result.output_json is None and result.output_ref is None
            assert result.attempts == 1
            continue
        output = result.output_json
        if result.output_ref is not None:
            output = await adapter._STORE.get(result.output_ref)
        assert json.loads(output) == {"text": result.query_id}


async def assert_child_layout(env, handle, session, expected_batches, probe):  # type: ignore[no-untyped-def]
    history = await handle.fetch_history()
    children = [
        event.start_child_workflow_execution_initiated_event_attributes
        for event in history.events
        if event.HasField("start_child_workflow_execution_initiated_event_attributes")
    ]
    parent_hash = hashlib.sha256(handle.id.encode()).hexdigest()
    expected_ids = [
        f"forkreduce:{parent_hash}:{session.prefix_hash}:batch:{index}" for index in range(len(expected_batches))
    ]
    assert [child.workflow_id for child in children] == expected_ids
    assert len(set(expected_ids)) == len(expected_batches)
    for child_id, batch in zip(expected_ids, expected_batches, strict=True):
        assert {query_id for query_id, workflow_id in probe.workflows.items() if workflow_id == child_id} == set(batch)
        child_results = await env.client.get_workflow_handle(
            child_id, result_type=list[adapter.QueryActivityOutput]
        ).result()
        assert [result.query_id for result in child_results] == batch
    return history


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("limit", [1, 2])
async def test_activity_concurrency_bounds_full_queries(plugin: bool, batch_size: int | None, limit: int) -> None:
    async def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        query_id = activity.info().activity_id.rsplit(":", 1)[1]
        await asyncio.sleep(0.08)
        return tool_result(info, "SchedulingAnswer", {"text": query_id})

    scripted = Scripted(handler)
    session = make_session(scripted.model())
    probe = ActivityProbe()
    config = adapter.MapQueriesInput(
        queries=queries_for(session),
        batch_size=batch_size,
        child_concurrency=1,
        activity_concurrency=limit,
    )
    async with running_map(config, probe, plugin=plugin) as (env, handle, _parent_queue, work_queue):
        results = await asyncio.wait_for(handle.result(), 30)
        await assert_outputs(results, session)
        assert probe.active == 0
        assert probe.peak == limit
        assert scripted.max_inflight == limit
        assert len(scripted.calls) == 6
        assert probe.queues == [work_queue] * 6
        if batch_size is not None:
            await assert_child_layout(env, handle, session, [["q0", "q1", "q2"], ["q3", "q4", "q5"]], probe)


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("batch_size", [None, 3])
async def test_task_queue_routes_activities_and_children(plugin: bool, batch_size: int | None) -> None:
    session = make_session(TestModel(custom_output_args={"text": "routed"}))
    probe = ActivityProbe()
    config = adapter.MapQueriesInput(queries=queries_for(session), batch_size=batch_size, activity_concurrency=2)
    async with running_map(config, probe, plugin=plugin, separate_queue=True) as (
        env,
        handle,
        parent_queue,
        work_queue,
    ):
        results = await asyncio.wait_for(handle.result(), 30)
        assert [result.query_id for result in results] == [f"q{index}" for index in range(6)]
        assert all(result.failure_kind is None for result in results)
        assert probe.queues == [work_queue] * 6
        assert parent_queue != work_queue
        parent_history = await handle.fetch_history()
        assert parent_history.events[0].workflow_execution_started_event_attributes.task_queue.name == parent_queue
        children = [
            event.start_child_workflow_execution_initiated_event_attributes
            for event in parent_history.events
            if event.HasField("start_child_workflow_execution_initiated_event_attributes")
        ]
        assert len(children) == (2 if batch_size else 0)
        activity_histories = []
        if batch_size:
            assert not any(
                event.HasField("activity_task_scheduled_event_attributes") for event in parent_history.events
            )
            for child in children:
                assert child.task_queue.name == work_queue
                history = await env.client.get_workflow_handle(child.workflow_id).fetch_history()
                assert history.events[0].workflow_execution_started_event_attributes.task_queue.name == work_queue
                activity_histories.append(history)
        else:
            activity_histories.append(parent_history)
        scheduled = [
            event.activity_task_scheduled_event_attributes
            for history in activity_histories
            for event in history.events
            if event.HasField("activity_task_scheduled_event_attributes")
        ]
        assert len(scheduled) == 6
        assert all(attrs.task_queue.name == work_queue for attrs in scheduled)
        assert len({attrs.activity_id for attrs in scheduled}) == 6
        for result in results:
            output = result.output_json
            if result.output_ref is not None:
                output = await adapter._STORE.get(result.output_ref)
            assert json.loads(output) == {"text": "routed"}


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize(
    ("warmup", "failed_first"),
    [("none", False), ("first", False), ("per_batch", False), ("first", True), ("per_batch", True)],
)
async def test_warmup_gates_full_activity_lifetimes_and_keeps_order(
    plugin: bool,
    batch_size: int | None,
    warmup: Literal["none", "first", "per_batch"],
    failed_first: bool,
) -> None:
    probe = ActivityProbe()

    async def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        query_id = activity.info().activity_id.rsplit(":", 1)[1]
        if query_id == "q0" and warmup == "none":
            await asyncio.wait_for(probe.completion("q1").wait(), 10)
        elif query_id == "q0" and warmup == "per_batch" and batch_size is not None:
            # A later query in the other batch must finish while this batch's first
            # query is still running: per-batch warmup is not a parent-wide gate.
            await asyncio.wait_for(probe.completion("q4").wait(), 10)
        else:
            await asyncio.sleep(0.08 if query_id == "q1" else 0.01)
        if failed_first and query_id == "q0":
            raise ModelHTTPError(401, "offline", body="synthetic permanent failure")
        return tool_result(info, "SchedulingAnswer", {"text": query_id})

    session = make_session(Scripted(handler).model())
    config = adapter.MapQueriesInput(
        queries=queries_for(session),
        batch_size=batch_size,
        child_concurrency=2,
        warmup=warmup,
        on_query_failure="return",
    )
    async with running_map(config, probe, plugin=plugin) as (env, handle, _parent_queue, _work_queue):
        results = await asyncio.wait_for(handle.result(), 30)
        await assert_outputs(results, session, failed_first=failed_first)
        assert probe.active == 0
        assert len(probe.trace) == 12
        position = {event: index for index, event in enumerate(probe.trace)}
        if warmup == "none":
            assert position[("finish", "q1")] < position[("finish", "q0")]
            assert position[("start", "q1")] < position[("finish", "q0")]
        elif warmup == "first" or batch_size is None:
            assert all(position[("finish", "q0")] < position[("start", f"q{index}")] for index in range(1, 6))
            assert position[("finish", "q2")] < position[("finish", "q1")]
        else:
            for first, rest in (("q0", ("q1", "q2")), ("q3", ("q4", "q5"))):
                assert all(position[("finish", first)] < position[("start", query_id)] for query_id in rest)
            assert position[("finish", "q4")] < position[("finish", "q0")]
        if batch_size is not None:
            batches = (
                [["q0"], ["q1", "q2", "q3"], ["q4", "q5"]]
                if warmup == "first"
                else [["q0", "q1", "q2"], ["q3", "q4", "q5"]]
            )
            history = await assert_child_layout(env, handle, session, batches, probe)
            if warmup == "first":
                starts = [
                    index
                    for index, event in enumerate(history.events)
                    if event.HasField("start_child_workflow_execution_initiated_event_attributes")
                ]
                first_completion = next(
                    index
                    for index, event in enumerate(history.events)
                    if event.HasField("child_workflow_execution_completed_event_attributes")
                )
                assert starts[0] < first_completion < min(starts[1:])


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("batch_size", [None, 3])
async def test_first_warmup_raise_does_not_release_gate(plugin: bool, batch_size: int | None) -> None:
    def handler(_messages, _info, _n):  # type: ignore[no-untyped-def]
        raise ModelHTTPError(401, "offline", body="synthetic permanent failure")

    scripted = Scripted(handler)
    session = make_session(scripted.model())
    probe = ActivityProbe()
    config = adapter.MapQueriesInput(
        queries=queries_for(session), batch_size=batch_size, warmup="first", on_query_failure="raise"
    )
    async with running_map(config, probe, plugin=plugin) as (_env, handle, _parent_queue, _work_queue):
        with pytest.raises(WorkflowFailureError):
            await asyncio.wait_for(handle.result(), 30)
        assert probe.trace == [("start", "q0"), ("finish", "q0")]
        assert probe.active == 0
        assert len(scripted.calls) == 1
        history = await handle.fetch_history()
        children = [
            event
            for event in history.events
            if event.HasField("start_child_workflow_execution_initiated_event_attributes")
        ]
        assert len(children) == (1 if batch_size else 0)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
@pytest.mark.parametrize("workflow_type", [adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow])
async def test_invalid_activity_concurrency_rejects_before_scheduling(  # type: ignore[no-untyped-def]
    limit, workflow_type
):
    session = make_session(TestModel(custom_output_args={"text": "unused"}))
    config = adapter.MapQueriesInput(queries=queries_for(session), activity_concurrency=limit)
    with pytest.raises(ApplicationError, match="activity_concurrency") as caught:
        await workflow_type().run(config)
    assert caught.value.non_retryable
