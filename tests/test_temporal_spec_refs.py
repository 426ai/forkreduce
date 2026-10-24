"""Referenced schemas, bounded generation, namespace safety and real replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
    from pydantic_ai.models.test import TestModel
    from tests.support import Scripted, text_result, tool_result
    from tests.test_temporal_runtime import sandbox_runner, server_options

    from forkreduce import OutputSpec, OutputValidationContext, Policy, build
    from forkreduce.contrib import temporal as adapter


class RecordingStore:
    def __init__(self):
        self.data = {}
        self.puts = []
        self.reads = []

    async def put(self, key, payload, *, context):
        self.puts.append((key, context))
        ref = f"{key}:{hashlib.sha256(payload).hexdigest()}"
        self.data[ref] = payload
        return ref

    async def get(self, ref, *, context):
        self.reads.append((ref, context))
        return self.data[ref]


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in ("_SESSIONS", "_FINGERPRINTS", "_FACTORIES"):
        monkeypatch.setattr(adapter, name, {})
    for name in ("_DEFAULT_FACTORY", "_STORE", "_PREFIX_STORE"):
        monkeypatch.setattr(adapter, name, None)


async def test_spec_writer_deduplicates_concurrent_values_and_isolates_namespaces():
    store = RecordingStore()
    writer = adapter.OutputSpecWriter(store)
    spec = OutputSpec("Value", {"type": "object", "properties": {"value": {"type": "integer"}}})
    refs = await asyncio.gather(*(writer.store(spec, ref_namespace="a/b") for _ in range(10)))
    assert refs == [refs[0]] * 10
    assert len(store.puts) == 1
    key, context = store.puts[0]
    assert key.startswith("a%2Fb/output-spec:")
    assert context.kind == "output_spec" and context.namespace == "a/b"
    raw = await store.get(refs[0].ref, context=context)
    assert json.loads(raw)["spec"]["json_schema"] == spec.json_schema
    other = await writer.store(spec, ref_namespace="other")
    assert other.ref != refs[0].ref
    assert json.loads(store.data[other.ref])["namespace"] == "other"
    changed = await writer.store(
        replace(spec, json_schema={"type": "object", "required": ["another"]}), ref_namespace="a/b"
    )
    assert changed.sha256 != refs[0].sha256
    assert json.loads(store.data[refs[0].ref])["spec"]["json_schema"] == spec.json_schema


@pytest.mark.parametrize("problem", ["missing", "digest", "name", "namespace"])
async def test_bad_spec_refs_fail_nonretryably_before_model(problem):
    store = RecordingStore()
    session = build(
        TestModel(), documents="Evidence", instructions="Answer", outputs=[], dynamic_outputs="generic_tool"
    )
    adapter.register_session(session)
    adapter.set_message_store(store)
    ref = await adapter.OutputSpecWriter(store).store(OutputSpec("Value", {"type": "object"}), ref_namespace="one")
    query = adapter.query_activity_input(session, query_id="q", tail="Question", output_spec=ref, ref_namespace="one")
    if problem == "missing":
        del store.data[ref.ref]
    elif problem == "digest":
        store.data[ref.ref] = b"{}"
    elif problem == "name":
        query.output_spec = replace(ref, name="Wrong")
        query.output_name = "Wrong"
    else:
        query.meta["ref_namespace"] = "two"
    with pytest.raises(ApplicationError) as caught:
        await ActivityEnvironment().run(adapter.execute_query_activity, query)
    assert caught.value.non_retryable
    if problem == "namespace":
        assert caught.value.type == "ref_namespace_mismatch"
        assert store.reads == []
    elif problem == "missing":
        assert "blob is missing" in str(caught.value)
    else:
        assert caught.value.type == "permanent"


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("mode", ["tools", "prompted"])
async def test_ref_specs_feedback_fresh_schema_switch_custom_layout_context_and_replay(plugin, mode):
    requests = Counter()
    seen = {}
    contexts = []
    store = RecordingStore()
    writer = adapter.OutputSpecWriter(store)
    first_ref = await writer.store(
        OutputSpec(
            "First",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        ),
        ref_namespace="specs",
    )
    second_ref = await writer.store(
        OutputSpec(
            "Second",
            {
                "type": "object",
                "properties": {"value": {"const": "done"}},
                "required": ["value"],
            },
        ),
        ref_namespace="specs",
    )

    def reply(messages, info, _number):
        qid = activity.info().activity_id.rsplit(":", 1)[-1]
        requests[qid] += 1
        seen.setdefault(qid, []).append(list(messages))
        value = ("invalid" if requests[qid] == 1 else 3) if qid == "root" else "done"
        return (
            tool_result(info, info.output_tools[0].name, {"value": value})
            if mode == "tools"
            else text_result({"value": value})
        )

    def validate(output, context):
        contexts.append(context)
        assert context.meta == {"ref_namespace": "specs", "synthetic": 7}
        return output

    scripted = Scripted(reply)

    def restore(frozen=None):
        return build(
            scripted.model(),
            documents=" prefix\x00 " if frozen is None else frozen.documents,
            instructions="Use application instructions.",
            outputs=[],
            output_mode=mode,
            dynamic_outputs="generic_tool" if mode == "tools" else "prompted",
            document_format="raw",
            message_layout="single",
            query_separator="|CUSTOM|" if frozen is None else frozen.query_separator,
            dynamic_output_instructions=False if frozen is None else frozen.dynamic_output_instructions,
            output_validator=validate,
            policy=Policy(retries=0, output_feedback_retries=1),
        )

    session = restore()
    adapter.set_message_store(store)
    adapter.set_prefix_store(store)
    adapter.set_default_session_factory(restore)
    prefix_ref = await adapter.freeze_prefix(session, store, app_meta={"ref_namespace": "specs"})
    query = adapter.query_activity_input(
        session,
        query_id="root",
        tail=" initial\t ",
        output_spec=first_ref,
        prefix_ref=prefix_ref,
        ref_namespace="specs",
        meta={"synthetic": 7},
    )
    hooks = []

    @activity.defn(name="synthetic.spec_review")
    async def review(data: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        hooks.append(data)
        assert isinstance(data.query.output_spec, adapter.OutputSpecRef)
        if data.round == 0:
            return adapter.AfterQueryDecision(
                "follow_up",
                follow_up_query=adapter.FollowUpQuery(
                    replace(data.query, output_spec=second_ref, output_name=second_ref.name, tail=" fresh\n "),
                    history="fresh",
                ),
            )
        return adapter.AfterQueryDecision("done")

    plugins = [PydanticAIPlugin()] if plugin else []
    async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
        converter = env.client.data_converter
        snapshot = await adapter._load_prefix(query)
        objects = [first_ref, snapshot, adapter.QueryLayout("|CUSTOM|", False), adapter.FollowUpQuery(query, "fresh")]
        hints = [adapter.OutputSpecRef, adapter.ConfiguredFrozenPrefix, adapter.QueryLayout, adapter.FollowUpQuery]
        objects.append(OutputValidationContext("context", 2, {"nested": {"scope": "synthetic"}}))
        hints.append(OutputValidationContext)
        assert await converter.decode(await converter.encode(objects), hints) == objects
        queue = f"spec-ref-{uuid4()}"
        workflows = [adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow]
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=workflows,
            activities=[adapter.run_query_activity, review],
            workflow_runner=sandbox_runner(),
        ):
            handle = await env.client.start_workflow(
                adapter.MapQueriesWorkflow.run,
                adapter.MapQueriesInput([query], batch_size=1, after_query_activity="synthetic.spec_review"),
                id=f"specs-{uuid4()}",
                task_queue=queue,
                execution_timeout=timedelta(seconds=30),
            )
            results = await handle.result()
            history = await handle.fetch_history()
            child_histories = [
                await env.client.get_workflow_handle(
                    e.start_child_workflow_execution_initiated_event_attributes.workflow_id
                ).fetch_history()
                for e in history.events
                if e.HasField("start_child_workflow_execution_initiated_event_attributes")
            ]
        for recorded in [history, *child_histories]:
            await Replayer(workflows=workflows, plugins=plugins, workflow_runner=sandbox_runner()).replay_workflow(
                recorded
            )
    result = results[0]
    assert result.query_id == "root#r1" and result.failure_kind is None
    assert json.loads(store.data[result.output_ref]) == {"value": "done"}
    assert requests == {"root": 2, "root#r1": 1}
    assert [(context.query_id, context.round) for context in contexts] == [("root", 0), ("root#r1", 1)]
    spec_reads = [(ref, ctx) for ref, ctx in store.reads if ctx.kind == "output_spec"]
    assert Counter(ref for ref, _ctx in spec_reads) == {first_ref.ref: 1, second_ref.ref: 1}
    assert all(ctx.namespace == "specs" for _ref, ctx in spec_reads)
    for qid, tail in (("root", " initial\t "), ("root#r1", " fresh\n ")):
        first = seen[qid][0]
        assert not any(isinstance(message, ModelResponse) for message in first)
        parts = [
            part.content
            for message in first
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        assert parts == [" prefix\x00 |CUSTOM|" + tail]
    assert [data.query.output_spec for data in hooks] == [first_ref, second_ref]


@pytest.mark.parametrize("plugin", [False, True])
async def test_generated_300_large_specs_never_enter_workflow_inputs(plugin):
    from examples.run_spec_refs_offline import run

    observed = await run(plugin=plugin)
    assert observed["queries"] == 300 and observed["model_requests"] == 301
    assert observed["workflow_input_max_bytes"] < 512 * 1024
    assert observed["activity_input_max_bytes"] < 512 * 1024
    assert observed["activity_result_max_bytes"] < 512 * 1024
    assert observed["replayed_histories"] == 1


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("original_instructions", [False, True])
async def test_worker_cannot_silently_change_frozen_tail_instructions(cached, original_instructions):
    store = RecordingStore()

    def make(instructions):
        return build(
            TestModel(),
            documents="Evidence",
            instructions="Answer",
            outputs=[],
            dynamic_outputs="generic_tool",
            dynamic_output_instructions=instructions,
        )

    original, wrong = make(original_instructions), make(not original_instructions)
    assert original.prefix_hash == wrong.prefix_hash
    ref = await adapter.freeze_prefix(original, store)
    adapter.set_prefix_store(store)
    if cached:
        adapter.register_session(wrong)
    else:
        adapter.set_default_session_factory(lambda _snapshot: wrong)
    query = adapter.query_activity_input(
        original,
        query_id="q",
        tail="Literal question",
        prefix_ref=ref,
        output_spec=OutputSpec("Value", {"type": "object"}),
    )
    with pytest.raises(ApplicationError, match="query layout") as caught:
        await ActivityEnvironment().run(adapter.execute_query_activity, query)
    assert caught.value.non_retryable
