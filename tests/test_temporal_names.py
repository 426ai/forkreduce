"""Registration namespaces through real offline workers, converters, and replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import FrozenInstanceError, asdict
from datetime import timedelta
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.converter import DataConverter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from pydantic_ai.exceptions import ModelHTTPError
    from temporalio.client import WorkflowFailureError
    from temporalio.contrib.pydantic import pydantic_data_converter
    from tests.support import Scripted, tool_result
    from tests.test_temporal_runtime import CoexistingAgentWorkflow, sandbox_runner, server_options

    from forkreduce import ActivityOptions, build
    from forkreduce.contrib import temporal as adapter
    from forkreduce.policy import Policy


class NamesAnswer(BaseModel):
    text: str


@workflow.defn(name="synthetic.names.custom_batch")
class NamesCustomBatchWorkflow:
    @workflow.run
    async def run(self, config: adapter.MapQueriesInput) -> list[adapter.QueryActivityOutput]:
        return await adapter.QueryBatchWorkflow().run(config)


@pytest.mark.parametrize("converter", [DataConverter.default, pydantic_data_converter])
@pytest.mark.parametrize("prefix", ["", "tenant.one/", "équipe:"])
async def test_temporal_names_roundtrip_and_registration_identity(converter: DataConverter, prefix: str) -> None:
    names = adapter.forkreduce_temporal_names(prefix=prefix)
    assert isinstance(names, adapter.TemporalNames)
    assert names.activity_name == prefix + "forkreduce.run_query"
    assert names.map_workflow_name == prefix + "forkreduce.map_queries"
    assert names.batch_workflow_name == prefix + "forkreduce.query_batch"
    with pytest.raises(FrozenInstanceError):
        names.prefix = "changed"  # type: ignore[misc]
    # Accessing registrations must not turn Python definitions into payload data.
    activities, workflows = names.activities, names.workflows
    encoded = await converter.encode([names])
    assert json.loads(encoded[0].data) == {"prefix": prefix}
    restored = (await converter.decode(encoded, [adapter.TemporalNames]))[0]
    assert restored == names
    if not prefix:
        assert activities == (adapter.run_query_activity,)
        assert workflows == (adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow)
        assert adapter.forkreduce_temporal_names() == adapter.TemporalNames()


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("batch_size", [None, 2])
async def test_default_and_prefixed_workers_names_results_failures_and_replay(
    plugin: bool, batch_size: int | None
) -> None:
    seen: list[tuple[str, str, str]] = []

    async def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        current = activity.info()
        query_id = current.activity_id.rsplit(":", 1)[1]
        seen.append((current.workflow_id, current.activity_type, query_id))
        if query_id == "fail":
            raise ModelHTTPError(status_code=401, model_name="offline", body="synthetic denied")
        await asyncio.sleep(0.01 * (3 - int(query_id[1:])))
        return tool_result(info, "NamesAnswer", {"text": query_id})

    session = build(
        Scripted(handler).model(),
        documents="Synthetic registration namespace evidence.",
        instructions="Use the registered output.",
        outputs=[NamesAnswer],
        warmup=False,
        policy=Policy(retries=0),
        activity_options=ActivityOptions(start_to_close=10, maximum_attempts=1),
    )
    prefix_hash = adapter.register_session(session)
    store = adapter.InMemoryMessageStore()
    adapter.set_message_store(store)
    defaults = adapter.forkreduce_temporal_names()
    prefixed = adapter.forkreduce_temporal_names(prefix="tenant.one/")
    plugins = [PydanticAIPlugin()] if plugin else []
    workflows = [*defaults.workflows, *prefixed.workflows]
    if plugin:
        workflows.append(CoexistingAgentWorkflow)
    runner = sandbox_runner()
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"temporal-names-{uuid4()}"
            histories = []
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=workflows,
                activities=[*defaults.activities, *prefixed.activities],
                workflow_runner=runner,
            ):
                for names in (defaults, prefixed):
                    for list_input in [False, True] if batch_size is None else [False]:
                        queries = [
                            adapter.query_activity_input(
                                session, query_id=f"q{i}", tail=f"Question {i}", output_name="NamesAnswer"
                            )
                            for i in range(3)
                        ]
                        config = adapter.MapQueriesInput(queries=queries, batch_size=batch_size)
                        handle = await env.client.start_workflow(
                            names.map_workflow_name,
                            queries if list_input else config,
                            result_type=list[adapter.QueryActivityOutput],
                            id=f"names-map-{uuid4()}",
                            task_queue=queue,
                            execution_timeout=timedelta(seconds=30),
                        )
                        results = await asyncio.wait_for(handle.result(), 30)
                        assert [result.query_id for result in results] == ["q0", "q1", "q2"]
                        for result in results:
                            output = (
                                await store.get(result.output_ref)
                                if result.output_ref is not None
                                else result.output_json
                            )
                            assert output is not None
                            assert json.loads(output) == {"text": result.query_id}
                            assert result.failure_kind is None
                        # The generated map copies configuration instead of mutating caller data.
                        assert config.activity_name == "forkreduce.run_query"
                        assert config.batch_workflow is None
                        history = await handle.fetch_history()
                        histories.append(history)
                        assert (
                            history.events[0].workflow_execution_started_event_attributes.workflow_type.name
                            == names.map_workflow_name
                        )
                        children = [
                            event.start_child_workflow_execution_initiated_event_attributes
                            for event in history.events
                            if event.HasField("start_child_workflow_execution_initiated_event_attributes")
                        ]
                        if batch_size is None:
                            assert children == []
                            activity_histories = [history]
                        else:
                            assert [child.workflow_type.name for child in children] == [names.batch_workflow_name] * 2
                            parent_hash = hashlib.sha256(handle.id.encode()).hexdigest()
                            assert [child.workflow_id for child in children] == [
                                f"forkreduce:{parent_hash}:{prefix_hash}:batch:{i}" for i in range(2)
                            ]
                            activity_histories = [
                                await env.client.get_workflow_handle(child.workflow_id).fetch_history()
                                for child in children
                            ]
                            histories.extend(activity_histories)
                        scheduled = [
                            event.activity_task_scheduled_event_attributes
                            for activity_history in activity_histories
                            for event in activity_history.events
                            if event.HasField("activity_task_scheduled_event_attributes")
                        ]
                        assert {entry.activity_type.name for entry in scheduled} == {names.activity_name}
                        assert {entry.activity_id for entry in scheduled} == {
                            f"forkreduce:{prefix_hash}:q{i}" for i in range(3)
                        }
                        execution_ids = {handle.id, *(child.workflow_id for child in children)}
                        assert {name for workflow_id, name, _ in seen if workflow_id in execution_ids} == {
                            names.activity_name
                        }
                    if batch_size is not None:
                        batch_config = adapter.MapQueriesInput(
                            queries=[
                                adapter.query_activity_input(
                                    session, query_id="q0", tail="Standalone batch", output_name="NamesAnswer"
                                )
                            ]
                        )
                        standalone = await env.client.start_workflow(
                            names.batch_workflow_name,
                            batch_config,
                            result_type=list[adapter.QueryActivityOutput],
                            id=f"names-standalone-batch-{uuid4()}",
                            task_queue=queue,
                            execution_timeout=timedelta(seconds=30),
                        )
                        batch_results = await asyncio.wait_for(standalone.result(), 30)
                        assert [result.query_id for result in batch_results] == ["q0"]
                        assert batch_results[0].output_ref is not None
                        assert json.loads(await store.get(batch_results[0].output_ref)) == {"text": "q0"}
                        assert batch_config.activity_name == "forkreduce.run_query"
                        batch_history = await standalone.fetch_history()
                        histories.append(batch_history)
                        assert (
                            batch_history.events[0].workflow_execution_started_event_attributes.workflow_type.name
                            == names.batch_workflow_name
                        )
                        assert [
                            event.activity_task_scheduled_event_attributes.activity_type.name
                            for event in batch_history.events
                            if event.HasField("activity_task_scheduled_event_attributes")
                        ] == [names.activity_name]
                    for return_failure in (False, True):
                        failing = adapter.query_activity_input(
                            session, query_id="fail", tail="Denied question", output_name="NamesAnswer"
                        )
                        config = adapter.MapQueriesInput(queries=[failing], batch_size=batch_size)
                        if return_failure:
                            config.on_query_failure = "return"
                        else:
                            assert config.on_query_failure == "raise"
                        handle = await env.client.start_workflow(
                            names.map_workflow_name,
                            config,
                            result_type=list[adapter.QueryActivityOutput],
                            id=f"names-failure-{uuid4()}",
                            task_queue=queue,
                            execution_timeout=timedelta(seconds=30),
                        )
                        if return_failure:
                            results = await asyncio.wait_for(handle.result(), 30)
                            assert len(results) == 1
                            assert results[0].query_id == "fail"
                            assert results[0].failure_kind == "permanent"
                            assert results[0].output_json is None and results[0].output_ref is None
                        else:
                            with pytest.raises(WorkflowFailureError):
                                await asyncio.wait_for(handle.result(), 30)
                        history = await handle.fetch_history()
                        histories.append(history)
                        for event in history.events:
                            if event.HasField("start_child_workflow_execution_initiated_event_attributes"):
                                child_id = event.start_child_workflow_execution_initiated_event_attributes.workflow_id
                                histories.append(await env.client.get_workflow_handle(child_id).fetch_history())
                if plugin:
                    official = await env.client.start_workflow(
                        CoexistingAgentWorkflow.run,
                        "Independent official durability execution on the same worker",
                        id=f"names-official-{uuid4()}",
                        task_queue=queue,
                        execution_timeout=timedelta(seconds=30),
                    )
                    assert await asyncio.wait_for(official.result(), 30) == "durable"
                    histories.append(await official.fetch_history())
            replayer = Replayer(workflows=workflows, plugins=plugins, workflow_runner=runner)
            for history in histories:
                await replayer.replay_workflow(history)
    finally:
        adapter.unregister_session(prefix_hash)
        adapter.set_message_store(None)


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("mode", ["direct", "batch", "custom"])
async def test_prefixed_hooks_and_custom_batches_keep_application_names(plugin: bool, mode: str) -> None:
    activity_names: list[str] = []
    hook_names: list[str] = []

    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        current = activity.info()
        activity_names.append(current.activity_type)
        return tool_result(info, "NamesAnswer", {"text": current.activity_id.rsplit(":", 1)[1]})

    @activity.defn(name="synthetic.names.after_query")
    async def after_query(payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        hook_names.append(activity.info().activity_type)
        if payload.round == 0:
            return adapter.AfterQueryDecision("follow_up", follow_up_tail="Second offline question")
        return adapter.AfterQueryDecision("done", result_meta={"application": [True, 2, None]})

    session = build(
        Scripted(handler).model(),
        documents="Synthetic hooks and custom batch namespace evidence.",
        instructions="Use the registered output.",
        outputs=[NamesAnswer],
        warmup=False,
        policy=Policy(retries=0),
        activity_options=ActivityOptions(start_to_close=10, maximum_attempts=1),
    )
    prefix_hash = adapter.register_session(session)
    store = adapter.InMemoryMessageStore()
    adapter.set_message_store(store)
    defaults = adapter.forkreduce_temporal_names()
    names = adapter.forkreduce_temporal_names(prefix="application:")
    plugins = [PydanticAIPlugin()] if plugin else []
    workflows = [*defaults.workflows, *names.workflows, NamesCustomBatchWorkflow]
    runner = sandbox_runner()
    query = adapter.query_activity_input(session, query_id="root", tail="First question", output_name="NamesAnswer")
    config = adapter.MapQueriesInput(
        queries=[query],
        batch_size=None if mode == "direct" else 1,
        batch_workflow="synthetic.names.custom_batch" if mode == "custom" else None,
        after_query_activity="synthetic.names.after_query",
    )
    original_config = asdict(config)
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"temporal-names-hooks-{uuid4()}"
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=workflows,
                activities=[*defaults.activities, *names.activities, after_query],
                workflow_runner=runner,
            ):
                handle = await env.client.start_workflow(
                    names.map_workflow_name,
                    config,
                    result_type=list[adapter.QueryActivityOutput],
                    id=f"names-hooks-{uuid4()}",
                    task_queue=queue,
                    execution_timeout=timedelta(seconds=30),
                )
                results = await asyncio.wait_for(handle.result(), 30)
                assert len(results) == 1 and results[0].query_id == "root#r1"
                assert results[0].output_ref is not None
                assert json.loads(await store.get(results[0].output_ref)) == {"text": "root#r1"}
                assert results[0].result_meta == {"application": [True, 2, None]}
                assert activity_names == ["application:forkreduce.run_query"] * 2
                assert hook_names == ["synthetic.names.after_query"] * 2
                assert asdict(config) == original_config
                history = await handle.fetch_history()
                histories = [history]
                children = [
                    event.start_child_workflow_execution_initiated_event_attributes
                    for event in history.events
                    if event.HasField("start_child_workflow_execution_initiated_event_attributes")
                ]
                if mode == "direct":
                    assert children == []
                else:
                    assert len(children) == 1
                    assert children[0].workflow_type.name == (
                        "synthetic.names.custom_batch" if mode == "custom" else "application:forkreduce.query_batch"
                    )
                    histories.append(await env.client.get_workflow_handle(children[0].workflow_id).fetch_history())
                scheduled = [
                    event.activity_task_scheduled_event_attributes.activity_type.name
                    for item in histories
                    for event in item.events
                    if event.HasField("activity_task_scheduled_event_attributes")
                ]
                assert scheduled == ["application:forkreduce.run_query", "synthetic.names.after_query"] * 2
            replayer = Replayer(workflows=workflows, plugins=plugins, workflow_runner=runner)
            for history in histories:
                await replayer.replay_workflow(history)
    finally:
        adapter.unregister_session(prefix_hash)
        adapter.set_message_store(None)
