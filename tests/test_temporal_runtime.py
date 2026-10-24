"""Offline models through real Temporal workers, converters, and replay."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import timedelta
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel
    from pydantic_ai import Agent
    from pydantic_ai.durable_exec.temporal import (
        PydanticAIPlugin,
        PydanticAIWorkflow,
        TemporalDurability,
    )
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from pydantic_ai.models.test import TestModel
    from temporalio.client import WorkflowFailureError
    from tests.support import Scripted, tool_result, user_text

    from forkreduce import ActivityOptions, Query, build
    from forkreduce.contrib.temporal import (
        InMemoryMessageStore,
        MapQueriesInput,
        MapQueriesWorkflow,
        QueryBatchWorkflow,
        freeze_prefix,
        query_activity_input,
        register_session,
        register_session_factory,
        run_query_activity,
        set_message_store,
        set_prefix_store,
        unregister_session,
    )
    from forkreduce.policy import Policy


class RuntimeAnswer(BaseModel):
    text: str


durable_agent = Agent(
    TestModel(custom_output_args={"text": "durable"}),
    output_type=RuntimeAnswer,
    name="forkreduce-runtime-coexistence",
    capabilities=[TemporalDurability()],
)


@workflow.defn
class CoexistingAgentWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [durable_agent]

    @workflow.run
    async def run(self, question: str) -> str:
        result = await durable_agent.run(question)
        return result.output.text


def server_options() -> dict[str, str]:
    """An explicit cached binary also supports network-isolated CI."""
    binary = os.environ.get("TEMPORAL_TEST_SERVER")
    return {"test_server_existing_path": binary} if binary else {}


def sandbox_runner() -> SandboxedWorkflowRunner:
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules("forkreduce", "pydantic_ai")
    )


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("model_kind", ["test", "function"])
async def test_map_queries_converter_and_replay(plugin: bool, model_kind: str) -> None:
    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        assert activity.in_activity()
        return tool_result(info, "RuntimeAnswer", {"text": "mapped"})

    model = TestModel(custom_output_args={"text": "mapped"}) if model_kind == "test" else Scripted(handler).model()
    session = build(
        model,
        documents=f"offline converter {plugin} {model_kind}",
        instructions="Answer using the registered output.",
        outputs=[RuntimeAnswer],
        policy=Policy(retries=0),
    )
    prefix = register_session(session)
    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"forkreduce-runtime-{uuid4()}"
            workflows = [MapQueriesWorkflow, CoexistingAgentWorkflow] if plugin else [MapQueriesWorkflow]
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=workflows,
                activities=[run_query_activity],
                workflow_runner=runner,
            ):
                payloads = [
                    query_activity_input(session, query_id=f"q{i}", tail=f"Question {i}", output_name="RuntimeAnswer")
                    for i in range(2)
                ]
                handle = await env.client.start_workflow(
                    MapQueriesWorkflow.run, payloads, id=f"map-{uuid4()}", task_queue=queue
                )
                results = await asyncio.wait_for(handle.result(), timeout=30)
                assert [json.loads(result.output_json or "null") for result in results] == [
                    {"text": "mapped"},
                    {"text": "mapped"},
                ]
                assert [result.query_id for result in results] == ["q0", "q1"]
                assert all(result.messages_json and "Question" in result.messages_json for result in results)
                history = await handle.fetch_history()
                if plugin:
                    assert (
                        await env.client.execute_workflow(
                            CoexistingAgentWorkflow.run,
                            "A separate durable agent on the same worker",
                            id=f"coexist-{uuid4()}",
                            task_queue=queue,
                            execution_timeout=timedelta(seconds=30),
                        )
                        == "durable"
                    )
            await Replayer(workflows=workflows, plugins=plugins, workflow_runner=runner).replay_workflow(history)
            if not plugin:
                await Replayer(
                    workflows=[MapQueriesWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner
                ).replay_workflow(history)
    finally:
        unregister_session(prefix)


@pytest.mark.parametrize("plugin", [False, True])
async def test_batched_references_bound_history_and_options(plugin: bool) -> None:
    document = "Synthetic frozen evidence. " * 80_000
    seen_attempts: list[int] = []

    async def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        assert document in user_text(messages)
        seen_attempts.append(activity.info().attempt)
        # Yield provider I/O so child overlap is exercised without a CPU-size deadline.
        await asyncio.sleep(0.01)
        return tool_result(info, "RuntimeAnswer", {"text": "stored result"})

    model = Scripted(handler).model()
    session = build(
        model,
        documents=document,
        instructions="Read the frozen evidence.",
        outputs=[RuntimeAnswer],
        policy=Policy(retries=0),
        activity_options=ActivityOptions(start_to_close=20, heartbeat_timeout=5, maximum_attempts=4),
    )
    store = InMemoryMessageStore()
    prefix = session.prefix_hash
    prefix_ref = await freeze_prefix(session, store)
    rebuilt: list[str] = []

    def factory(frozen):  # type: ignore[no-untyped-def]
        rebuilt.append(frozen.prefix_hash)
        return build(
            model,
            documents=frozen.documents,
            instructions=frozen.instructions,
            outputs=[RuntimeAnswer],
            output_mode=frozen.output_mode,
            policy=Policy(retries=0),
        )

    register_session_factory(prefix, factory)
    set_prefix_store(store)
    set_message_store(store)
    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"forkreduce-batch-{uuid4()}"
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=[MapQueriesWorkflow, QueryBatchWorkflow],
                activities=[run_query_activity],
                workflow_runner=runner,
                max_heartbeat_throttle_interval=timedelta(milliseconds=40),
                default_heartbeat_throttle_interval=timedelta(milliseconds=40),
            ):
                payload = MapQueriesInput(
                    queries=[
                        query_activity_input(
                            session,
                            query_id=f"q{i}",
                            tail=f"Question {i}",
                            output_name="RuntimeAnswer",
                            prefix_ref=prefix_ref,
                        )
                        for i in range(6)
                    ],
                    batch_size=2,
                    child_concurrency=2,
                    activity_options=ActivityOptions(start_to_close=10, maximum_attempts=1),
                )
                handle = await env.client.start_workflow(
                    MapQueriesWorkflow.run, payload, id=f"batch-{uuid4()}", task_queue=queue
                )
                results = await asyncio.wait_for(handle.result(), timeout=30)
                assert [r.query_id for r in results] == [f"q{i}" for i in range(6)]
                assert seen_attempts == [1] * 6
                assert rebuilt and set(rebuilt) == {prefix}
                for result in results:
                    assert result.messages_json is None and result.messages_ref
                    assert len(await store.get(result.messages_ref)) > 2_000_000
                    assert result.output_json is None and result.output_ref
                    assert json.loads(await store.get(result.output_ref)) == {"text": "stored result"}
                parent_history = await handle.fetch_history()
                active = maximum = 0
                children = []
                for event in parent_history.events:
                    if event.HasField("start_child_workflow_execution_initiated_event_attributes"):
                        active += 1
                        maximum = max(maximum, active)
                        attrs = event.start_child_workflow_execution_initiated_event_attributes
                        assert prefix in attrs.workflow_id
                        children.append(attrs.workflow_id)
                    elif event.HasField("child_workflow_execution_completed_event_attributes"):
                        active -= 1
                assert len(children) == len(set(children)) == 3
                assert maximum == 2 and active == 0
                assert len(parent_history.to_json().encode()) < 64 * 1024
                assert not any(e.HasField("activity_task_scheduled_event_attributes") for e in parent_history.events)
                for child_id in children:
                    child_history = await env.client.get_workflow_handle(child_id).fetch_history()
                    assert len(child_history.to_json().encode()) < 64 * 1024
                    for event in child_history.events:
                        if event.HasField("activity_task_scheduled_event_attributes"):
                            attrs = event.activity_task_scheduled_event_attributes
                            assert attrs.start_to_close_timeout.ToTimedelta() == timedelta(seconds=10)
                            assert attrs.retry_policy.maximum_attempts == 1
                            assert all(len(p.data) < 16 * 1024 for p in attrs.input.payloads)
                    await Replayer(
                        workflows=[QueryBatchWorkflow], plugins=plugins, workflow_runner=runner
                    ).replay_workflow(child_history)
            await Replayer(
                workflows=[MapQueriesWorkflow, QueryBatchWorkflow], plugins=plugins, workflow_runner=runner
            ).replay_workflow(parent_history)
    finally:
        unregister_session(prefix)
        set_prefix_store(None)
        set_message_store(None)


async def test_recipe_enforces_two_total_attempts() -> None:
    attempts: list[int] = []

    def handler(_messages, _info, _n):  # type: ignore[no-untyped-def]
        attempts.append(activity.info().attempt)
        raise ModelHTTPError(status_code=503, model_name="offline", body="synthetic unavailable")

    session = build(
        Scripted(handler).model(),
        documents="retry recipe",
        instructions="Answer.",
        outputs=[RuntimeAnswer],
        policy=Policy(retries=5),
        activity_options=ActivityOptions(
            start_to_close=180,
            heartbeat_timeout=30,
            schedule_to_close=400,
            maximum_attempts=2,
            retry_backoff=0.01,
            max_backoff=0.05,
            backoff_coefficient=3,
        ),
    )
    prefix = register_session(session)
    try:
        async with await WorkflowEnvironment.start_time_skipping(**server_options()) as env:
            queue = f"forkreduce-retry-{uuid4()}"
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=[MapQueriesWorkflow],
                activities=[run_query_activity],
                workflow_runner=sandbox_runner(),
            ):
                handle = await env.client.start_workflow(
                    MapQueriesWorkflow.run,
                    [query_activity_input(session, query_id="fail", tail="Question", output_name="RuntimeAnswer")],
                    id=f"retry-{uuid4()}",
                    task_queue=queue,
                )
                with pytest.raises(WorkflowFailureError):
                    await asyncio.wait_for(handle.result(), timeout=30)
                assert attempts == [1, 2]
                history = await handle.fetch_history()
                scheduled = [
                    event.activity_task_scheduled_event_attributes
                    for event in history.events
                    if event.HasField("activity_task_scheduled_event_attributes")
                ]
                assert len(scheduled) == 1
                assert scheduled[0].start_to_close_timeout.ToTimedelta() == timedelta(minutes=3)
                assert scheduled[0].heartbeat_timeout.ToTimedelta() == timedelta(seconds=30)
                assert scheduled[0].retry_policy.maximum_attempts == 2
                assert scheduled[0].schedule_to_close_timeout.ToTimedelta() == timedelta(seconds=400)
                assert scheduled[0].retry_policy.initial_interval.ToTimedelta() == timedelta(milliseconds=10)
                assert scheduled[0].retry_policy.maximum_interval.ToTimedelta() == timedelta(milliseconds=50)
                assert scheduled[0].retry_policy.backoff_coefficient == 3
    finally:
        unregister_session(prefix)


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("forced", [False, True])
async def test_auto_snapshot_mode_survives_plugin_children_and_replay(plugin: bool, forced: bool) -> None:
    model = TestModel(
        custom_output_args={"text": "auto restored"} if forced else None,
        custom_output_text=None if forced else '{"text":"auto restored"}',
        profile={"supports_forced_tool_choice": forced},
    )
    session = build(
        model,
        documents="Auto mode frozen evidence.",
        instructions="Answer from evidence.",
        outputs=[RuntimeAnswer],
        on_unforced_tools="error",
    )
    expected = "tools" if forced else "prompted"
    store = InMemoryMessageStore()
    ref = await freeze_prefix(session, store)

    def restore(frozen):  # type: ignore[no-untyped-def]
        assert frozen.resolved_output_mode == expected
        return build(
            model,
            documents=frozen.documents,
            instructions=frozen.instructions,
            outputs=[RuntimeAnswer],
            output_mode=frozen.output_mode,
            on_unforced_tools=frozen.on_unforced_tools,
        )

    register_session_factory(session.prefix_hash, restore)
    set_prefix_store(store)
    set_message_store(store)
    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"forkreduce-auto-{uuid4()}"
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=[MapQueriesWorkflow, QueryBatchWorkflow],
                activities=[run_query_activity],
                workflow_runner=runner,
            ):
                handle = await env.client.start_workflow(
                    MapQueriesWorkflow.run,
                    MapQueriesInput(
                        queries=[
                            query_activity_input(
                                session,
                                query_id="q1",
                                tail="Question",
                                output_name="RuntimeAnswer",
                                prefix_ref=ref,
                            )
                        ],
                        batch_size=1,
                        child_concurrency=1,
                    ),
                    id=f"auto-{uuid4()}",
                    task_queue=queue,
                )
                results = await asyncio.wait_for(handle.result(), timeout=30)
                assert results[0].resolved_output_mode == expected
                assert results[0].output_ref and results[0].output_json is None
                assert json.loads(await store.get(results[0].output_ref)) == {"text": "auto restored"}
                history = await handle.fetch_history()
                child_id = next(
                    e.start_child_workflow_execution_initiated_event_attributes.workflow_id
                    for e in history.events
                    if e.HasField("start_child_workflow_execution_initiated_event_attributes")
                )
                child_history = await env.client.get_workflow_handle(child_id).fetch_history()
            replayer = Replayer(
                workflows=[MapQueriesWorkflow, QueryBatchWorkflow],
                plugins=plugins,
                workflow_runner=runner,
            )
            await replayer.replay_workflow(history)
            await replayer.replay_workflow(child_history)
    finally:
        unregister_session(session.prefix_hash)
        set_prefix_store(None)
        set_message_store(None)


@pytest.mark.parametrize("plugin", [False, True])
async def test_periodic_heartbeat_protects_awaiting_provider(plugin: bool) -> None:
    attempts = []

    async def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        attempts.append(activity.info().attempt)
        # Test liveness independently of large-document CPU hydration.
        await asyncio.sleep(0.65)
        return tool_result(info, "RuntimeAnswer", {"text": "heartbeat survived"})

    session = build(
        Scripted(handler).model(),
        documents="Small frozen evidence.",
        instructions="Answer.",
        outputs=[RuntimeAnswer],
        activity_options=ActivityOptions(start_to_close=10, heartbeat_timeout=0.2, maximum_attempts=1),
    )
    prefix = register_session(session)
    plugins = [PydanticAIPlugin()] if plugin else []
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"forkreduce-heartbeat-{uuid4()}"
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=[MapQueriesWorkflow],
                activities=[run_query_activity],
                workflow_runner=sandbox_runner(),
                max_heartbeat_throttle_interval=timedelta(milliseconds=40),
                default_heartbeat_throttle_interval=timedelta(milliseconds=40),
            ):
                results = await env.client.execute_workflow(
                    MapQueriesWorkflow.run,
                    [
                        query_activity_input(
                            session, query_id="q", tail="Wait for provider", output_name="RuntimeAnswer"
                        )
                    ],
                    id=f"heartbeat-{uuid4()}",
                    task_queue=queue,
                    execution_timeout=timedelta(seconds=30),
                )
                assert attempts == [1]
                assert json.loads(results[0].output_json) == {"text": "heartbeat survived"}
    finally:
        unregister_session(prefix)


class SlowReferenceStore(InMemoryMessageStore):
    """Delay one real storage boundary beyond the activity heartbeat deadline."""

    def __init__(self, phase: str) -> None:
        super().__init__()
        self.phase = phase
        self.waits: list[tuple[str, int]] = []

    async def pause(self, phase: str) -> None:
        if phase == self.phase and activity.in_activity():
            self.waits.append((phase, activity.info().attempt))
            await asyncio.sleep(0.65)

    async def get(self, ref: str) -> bytes:
        await self.pause("prefix_read" if ref.startswith("prefix:") else "history_read")
        return await super().get(ref)

    async def put(self, key: str, payload: bytes) -> str:
        if not key.startswith("prefix:"):
            await self.pause("output_write" if key.startswith("output:") else "messages_write")
        return await super().put(key, payload)


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("phase", ["prefix_read", "history_read", "messages_write", "output_write", "factory"])
async def test_heartbeat_covers_reference_io_and_async_factory(plugin: bool, phase: str) -> None:
    model_attempts: list[int] = []
    factory_attempts: list[int] = []

    def handler(messages, info, _n):  # type: ignore[no-untyped-def]
        if activity.in_activity():
            model_attempts.append(activity.info().attempt)
            assert "Final question" in user_text(messages)
            if phase == "history_read":
                assert "Prior question" in user_text(messages)
        return tool_result(info, "RuntimeAnswer", {"text": "persisted after slow I/O"})

    model = Scripted(handler).model()
    session = build(
        model,
        documents=f"Small frozen evidence for {phase}.",
        instructions="Answer from evidence.",
        outputs=[RuntimeAnswer],
        activity_options=ActivityOptions(start_to_close=10, heartbeat_timeout=0.2, maximum_attempts=1),
    )
    store = SlowReferenceStore(phase)
    prefix_ref = await freeze_prefix(session, store)
    history_ref = None
    if phase == "history_read":
        prior = await session.ask(Query(tail="Prior question", output=RuntimeAnswer))
        assert prior.failure is None
        history_ref = await store.put("history", ModelMessagesTypeAdapter.dump_json(prior.messages))

    async def restore(frozen):  # type: ignore[no-untyped-def]
        factory_attempts.append(activity.info().attempt)
        if phase == "factory":
            await asyncio.sleep(0.65)
        return build(
            model,
            documents=frozen.documents,
            instructions=frozen.instructions,
            outputs=[RuntimeAnswer],
            output_mode=frozen.output_mode,
        )

    register_session_factory(session.prefix_hash, restore)
    set_prefix_store(store)
    set_message_store(store)
    plugins = [PydanticAIPlugin()] if plugin else []
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"forkreduce-slow-reference-{uuid4()}"
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=[MapQueriesWorkflow, QueryBatchWorkflow],
                activities=[run_query_activity],
                workflow_runner=sandbox_runner(),
                max_heartbeat_throttle_interval=timedelta(milliseconds=40),
                default_heartbeat_throttle_interval=timedelta(milliseconds=40),
            ):
                results = await env.client.execute_workflow(
                    MapQueriesWorkflow.run,
                    MapQueriesInput(
                        queries=[
                            query_activity_input(
                                session,
                                query_id="q",
                                tail="Final question",
                                output_name="RuntimeAnswer",
                                prefix_ref=prefix_ref,
                                history_ref=history_ref,
                            )
                        ],
                        batch_size=1,
                        child_concurrency=1,
                    ),
                    id=f"slow-reference-{uuid4()}",
                    task_queue=queue,
                    execution_timeout=timedelta(seconds=30),
                )
                result = results[0]
                assert result.output_json is None and result.output_ref
                assert result.messages_json is None and result.messages_ref
                assert json.loads(await store.get(result.output_ref)) == {"text": "persisted after slow I/O"}
                messages = ModelMessagesTypeAdapter.validate_json(await store.get(result.messages_ref))
                assert "Final question" in user_text(messages)
                assert model_attempts == [1]
                assert factory_attempts == [1]
                assert store.waits == ([] if phase == "factory" else [(phase, 1)])
    finally:
        unregister_session(session.prefix_hash)
        set_prefix_store(None)
        set_message_store(None)
