"""Per-question JSON schemas through reference workers and both converters."""

from __future__ import annotations

import copy
import json
from datetime import timedelta
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from tests.support import Scripted, text_result, tool_result
    from tests.test_output_feedback import feedback_text
    from tests.test_temporal_runtime import sandbox_runner, server_options

    from forkreduce import OutputSpec, Policy, build
    from forkreduce.contrib import temporal as adapter


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_SESSIONS", "_FINGERPRINTS", "_FACTORIES"):
        monkeypatch.setattr(adapter, name, {})
    for name in ("_DEFAULT_FACTORY", "_STORE", "_PREFIX_STORE"):
        monkeypatch.setattr(adapter, name, None)


def output_specs() -> list[OutputSpec]:
    return [
        OutputSpec(
            "Label",
            {
                "type": "object",
                "properties": {"value": {"type": "string", "minLength": 2}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        OutputSpec(
            "Total",
            {
                "$defs": {"positive": {"type": "integer", "minimum": 1}},
                "type": "object",
                "properties": {"value": {"anyOf": [{"$ref": "#/$defs/positive"}, {"type": "null"}]}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
    ]


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("factory", [False, True])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("hook", [False, True])
async def test_dynamic_schemas_keep_one_prefix_and_replay(
    plugin: bool, mode: str, factory: bool, batch: bool, hook: bool
) -> None:
    observations = {"label": [], "total": []}
    seen_hooks = []

    def reply(messages, info, _number):  # type: ignore[no-untyped-def]
        query_id = activity.info().activity_id.rsplit(":", 1)[-1]
        observations[query_id].append((copy.deepcopy(messages), copy.deepcopy(info)))
        payload = {"value": "x" if len(observations[query_id]) == 1 else "OK" if query_id == "label" else 7}
        return tool_result(info, info.output_tools[0].name, payload) if mode == "tools" else text_result(payload)

    scripted = Scripted(reply)

    def restore(snapshot=None):  # type: ignore[no-untyped-def]
        return build(
            scripted.model(),
            outputs=[],
            documents="Synthetic dynamic evidence." if snapshot is None else snapshot.documents,
            instructions="Answer.",
            output_mode="auto",
            warmup=False,
            dynamic_outputs=("generic_tool" if mode == "tools" else "prompted")
            if snapshot is None
            else snapshot.dynamic_outputs,
            policy=Policy(
                retries=0, output_feedback_retries=1 if snapshot is None else snapshot.output_feedback_retries
            ),
        )

    session = restore()
    store = adapter.InMemoryMessageStore()
    adapter.set_message_store(store)
    prefix_ref = None
    if factory:
        prefix_ref = await adapter.freeze_prefix(
            session, store, app_meta={"ref_namespace": "dynamic", "nested": {"labels": ["synthetic"]}}
        )
        adapter.set_prefix_store(store)
        adapter.set_default_session_factory(restore)
    else:
        adapter.register_session(session)

    @activity.defn(name="synthetic.dynamic_review")
    async def review(data: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        seen_hooks.append(data)
        assert data.query.output_spec is not None
        return adapter.AfterQueryDecision("done", result_meta={"schema": data.query.output_spec.name})

    specs = output_specs()
    queries = [
        adapter.query_activity_input(
            session,
            query_id=name,
            tail="Question",
            output_spec=spec,
            prefix_ref=prefix_ref,
            ref_namespace="dynamic" if factory else None,
        )
        for name, spec in zip(observations, specs, strict=True)
    ]
    assert queries[0].prefix_hash == queries[1].prefix_hash
    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
        converter = env.client.data_converter
        snapshot_ref = await adapter.freeze_prefix(session, store, app_meta={"nested": {"labels": ["synthetic"]}})
        snapshot = adapter.FrozenPrefix(**json.loads(await store.get(snapshot_ref)))
        values = [specs[0], queries[0], snapshot, adapter.AfterQueryDecision("follow_up", follow_up_query=queries[1])]
        hints = [OutputSpec, adapter.QueryActivityInput, adapter.FrozenPrefix, adapter.AfterQueryDecision]
        assert await converter.decode(await converter.encode(values), hints) == values
        queue = f"dynamic-{uuid4()}"
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow],
            activities=[adapter.run_query_activity, review],
            workflow_runner=runner,
        ):
            handle = await env.client.start_workflow(
                adapter.MapQueriesWorkflow.run,
                adapter.MapQueriesInput(
                    queries=queries,
                    batch_size=2 if batch else None,
                    on_query_failure="return",
                    after_query_activity="synthetic.dynamic_review" if hook else None,
                ),
                id=f"dynamic-{uuid4()}",
                task_queue=queue,
                execution_timeout=timedelta(seconds=30),
            )
            results = await handle.result()
            assert [r.output_type for r in results] == ["Label", "Total"]
            assert all(r.failure_kind is None and r.feedback_retries_used == 1 for r in results)
            bodies = [json.loads(await store.get(r.output_ref) if r.output_ref else r.output_json) for r in results]
            assert bodies == [{"value": "OK"}, {"value": 7}]
            assert [r.result_meta for r in results] == (
                [{"schema": "Label"}, {"schema": "Total"}] if hook else [None, None]
            )
            if hook:
                assert {item.query.query_id: item.query.output_spec for item in seen_hooks} == dict(
                    zip(observations, specs, strict=True)
                )
                assert all(item.output.output_ref and item.output.messages_ref for item in seen_hooks)
            first_tools = observations["label"][0][1].output_tools
            for rows in observations.values():
                assert len(rows) == 2
                assert feedback_text(rows[1][0])
                assert all(row[0][0].parts[0].content == session.rendered_documents for row in rows)
                assert all(row[1].output_tools == first_tools for row in rows)
            history = await handle.fetch_history()
            children = [
                await env.client.get_workflow_handle(
                    e.start_child_workflow_execution_initiated_event_attributes.workflow_id
                ).fetch_history()
                for e in history.events
                if e.HasField("start_child_workflow_execution_initiated_event_attributes")
            ]
        replayer = Replayer(
            workflows=[adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow],
            plugins=plugins,
            workflow_runner=runner,
        )
        await replayer.replay_workflow(history)
        for child in children:
            await replayer.replay_workflow(child)


def test_dynamic_payload_copies_schema_and_rejects_name_disagreement() -> None:
    from pydantic_ai.models.test import TestModel

    session = build(TestModel(), outputs=[], documents="Synthetic", instructions="Answer.", output_mode="prompted")
    spec = output_specs()[0]
    payload = adapter.query_activity_input(session, query_id="q", tail="Question", output_spec=spec)
    spec.json_schema["required"].append("later")
    assert payload.output_spec.json_schema["required"] == ["value"]
    with pytest.raises(ValueError, match="match"):
        adapter.query_activity_input(session, query_id="q", tail="Question", output_name="different", output_spec=spec)
