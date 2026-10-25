"""Frozen raw bytes and combined user prompts through workers and repairs."""

from __future__ import annotations

import copy
import json
from datetime import timedelta
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
    from tests.support import Scripted, text_result, tool_result
    from tests.test_output_feedback import feedback_text
    from tests.test_temporal_runtime import sandbox_runner, server_options

    from forkreduce import QUERY_SEPARATOR, Policy, Query, build
    from forkreduce.contrib import temporal as adapter


class LayoutAnswer(BaseModel):
    """A synthetic layout answer."""

    text: str


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_SESSIONS", "_FINGERPRINTS", "_FACTORIES"):
        monkeypatch.setattr(adapter, name, {})
    for name in ("_DEFAULT_FACTORY", "_STORE", "_PREFIX_STORE"):
        monkeypatch.setattr(adapter, name, None)


def initial_user_parts(messages):  # type: ignore[no-untyped-def]
    parts = []
    for message in messages:
        if isinstance(message, ModelResponse):
            break
        if isinstance(message, ModelRequest):
            parts.extend(part.content for part in message.parts if isinstance(part, UserPromptPart))
    return parts


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("document_format", ["blocks", "raw"])
@pytest.mark.parametrize("message_layout", ["split", "single"])
@pytest.mark.parametrize("mode", ["tools", "prompted"])
async def test_frozen_layout_feedback_and_repair_replay(
    plugin: bool, document_format: str, message_layout: str, mode: str
) -> None:
    original_text = "  α\x00beta\n"
    observed = {"root": [], "root#r1": []}
    hooks = []

    def reply(messages, info, _number):  # type: ignore[no-untyped-def]
        query_id = activity.info().activity_id.rsplit(":", 1)[-1]
        observed[query_id].append(copy.deepcopy(messages))
        payload = (
            {"wrong": "synthetic"} if query_id == "root" and len(observed[query_id]) == 1 else {"text": "accepted"}
        )
        return tool_result(info, "LayoutAnswer", payload) if mode == "tools" else text_result(payload)

    scripted = Scripted(reply)

    def restore(snapshot=None):  # type: ignore[no-untyped-def]
        return build(
            scripted.model(),
            outputs=[LayoutAnswer],
            instructions="Answer.",
            documents=original_text if snapshot is None else snapshot.documents,
            output_mode=mode,
            document_format=document_format if snapshot is None else snapshot.document_format,
            message_layout=message_layout if snapshot is None else snapshot.message_layout,
            policy=Policy(retries=0, output_feedback_retries=1),
            warmup=False,
        )

    session = restore()
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store, app_meta={"ref_namespace": "layouts"})
    adapter.set_prefix_store(store)
    adapter.set_message_store(store)
    adapter.set_default_session_factory(restore)

    @activity.defn(name="synthetic.layout_review")
    async def review(data: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        hooks.append(data)
        if data.round == 0:
            return adapter.AfterQueryDecision("follow_up", follow_up_tail="Revise.")
        return adapter.AfterQueryDecision("done", result_meta={"round": data.round, "format": document_format})

    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
        snapshot = adapter.FrozenPrefix(**json.loads(await store.get(ref)))
        converter = env.client.data_converter
        assert (await converter.decode(await converter.encode([snapshot]), [adapter.FrozenPrefix])) == [snapshot]
        assert snapshot.document_format == document_format and snapshot.message_layout == message_layout
        queue = f"layouts-{uuid4()}"
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
                    queries=[
                        adapter.query_activity_input(
                            session,
                            query_id="root",
                            tail="Question.",
                            output_name="LayoutAnswer",
                            prefix_ref=ref,
                            ref_namespace="layouts",
                        )
                    ],
                    batch_size=1,
                    after_query_activity="synthetic.layout_review",
                    max_rounds=1,
                    on_query_failure="return",
                ),
                id=f"layouts-{uuid4()}",
                task_queue=queue,
                execution_timeout=timedelta(seconds=30),
            )
            result = (await handle.result())[0]
            assert result.failure_kind is None and result.query_id == "root#r1"
            assert result.result_meta == {"round": 1, "format": document_format}
            assert json.loads(await store.get(result.output_ref)) == {"text": "accepted"}
            assert [item.output.feedback_retries_used for item in hooks] == [1, 0]
            assert len(observed["root"]) == 2 and len(observed["root#r1"]) == 1
            assert feedback_text(observed["root"][1])
            tail = session.render_tail(Query(id="root", tail="Question.", output=LayoutAnswer))
            first = initial_user_parts(observed["root"][0])
            expected = (
                [session.rendered_documents, tail]
                if message_layout == "split"
                else [session.rendered_documents + QUERY_SEPARATOR + tail]
            )
            assert first == expected
            for messages in [*observed["root"], *observed["root#r1"]]:
                assert session.starts_with_prefix(messages)
                assert initial_user_parts(messages) == first
                user_text = "\n".join(
                    part.content
                    for m in messages
                    if isinstance(m, ModelRequest)
                    for part in m.parts
                    if isinstance(part, UserPromptPart)
                )
                assert user_text.count(session.rendered_documents) == 1
            if document_format == "raw":
                assert session.rendered_documents.encode() == original_text.encode()
                assert (
                    "[document" not in session.rendered_documents
                    and "===END_FIXED_DOCUMENTS===" not in session.rendered_documents
                )
            else:
                assert "\x00" not in session.rendered_documents
            history = await handle.fetch_history()
            children = [
                await env.client.get_workflow_handle(
                    e.start_child_workflow_execution_initiated_event_attributes.workflow_id
                ).fetch_history()
                for e in history.events
                if e.HasField("start_child_workflow_execution_initiated_event_attributes")
            ]
        replayer = Replayer(
            workflows=[adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow], plugins=plugins, workflow_runner=runner
        )
        await replayer.replay_workflow(history)
        for child in children:
            await replayer.replay_workflow(child)


@pytest.mark.parametrize("document_format,message_layout", [("raw", "split"), ("blocks", "single"), ("raw", "single")])
async def test_factory_cannot_restore_different_layout(document_format: str, message_layout: str) -> None:
    from pydantic_ai.models.test import TestModel

    original = build(
        TestModel(),
        documents="Synthetic",
        instructions="Answer.",
        outputs=[LayoutAnswer],
        document_format=document_format,
        message_layout=message_layout,
    )
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(original, store)
    adapter.set_prefix_store(store)
    adapter.set_default_session_factory(
        lambda snapshot: build(
            TestModel(),
            documents=snapshot.documents,
            instructions=snapshot.instructions,
            outputs=[LayoutAnswer],
        )
    )
    with pytest.raises(ApplicationError) as caught:
        await ActivityEnvironment().run(
            adapter.execute_query_activity,
            adapter.query_activity_input(
                original,
                query_id="q",
                tail="Question",
                output_name="LayoutAnswer",
                prefix_ref=ref,
            ),
        )
    assert caught.value.non_retryable is True
