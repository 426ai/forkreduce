"""Worker-local validation feedback through both converters and replay."""

from __future__ import annotations

import json
from datetime import timedelta
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel
    from pydantic_ai import ModelRetry
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from tests.support import Scripted, text_result, tool_result
    from tests.test_output_feedback import feedback_text
    from tests.test_temporal_runtime import sandbox_runner, server_options

    from forkreduce import Policy, build
    from forkreduce.contrib import temporal as adapter


class FeedbackAnswer(BaseModel):
    text: str


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_SESSIONS", "_FINGERPRINTS", "_FACTORIES"):
        monkeypatch.setattr(adapter, name, {})
    for name in ("_DEFAULT_FACTORY", "_STORE", "_PREFIX_STORE"):
        monkeypatch.setattr(adapter, name, None)


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("factory", [False, True])
@pytest.mark.parametrize("mode", ["tools", "prompted"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_worker_feedback_reuses_prefix_and_replays(
    plugin: bool, factory: bool, mode: str, batch: bool, asynchronous: bool
) -> None:
    attempts: list[int] = []

    def validator(output: FeedbackAnswer) -> FeedbackAnswer:
        if output.text == "invalid":
            raise ModelRetry("synthetic-validator-feedback")
        return output

    async def async_validator(output: FeedbackAnswer) -> FeedbackAnswer:
        return validator(output)

    def handler(_messages, info, n):  # type: ignore[no-untyped-def]
        attempts.append(activity.info().attempt)
        data = {"text": "invalid" if n == 1 else "repaired"}
        return tool_result(info, "FeedbackAnswer", data) if mode == "tools" else text_result(data)

    scripted = Scripted(handler)

    def make_session(snapshot=None):  # type: ignore[no-untyped-def]
        return build(
            scripted.model(),
            documents="Synthetic feedback evidence." if snapshot is None else snapshot.documents,
            instructions="Answer." if snapshot is None else snapshot.instructions,
            outputs=[FeedbackAnswer],
            output_mode=mode,
            warmup=False,
            policy=Policy(
                retries=0, output_feedback_retries=1 if snapshot is None else snapshot.output_feedback_retries
            ),
            output_validator=async_validator if asynchronous else validator,
        )

    session = make_session()
    store = adapter.InMemoryMessageStore()
    adapter.set_message_store(store if batch else None)
    prefix_ref = None
    if factory:
        prefix_ref = await adapter.freeze_prefix(session, store, app_meta={"ref_namespace": "synthetic"})
        adapter.set_prefix_store(store)
        adapter.set_default_session_factory(make_session)
    else:
        adapter.register_session(session)
    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
        queue = f"feedback-{uuid4()}"
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow],
            activities=[adapter.run_query_activity],
            workflow_runner=runner,
        ):
            handle = await env.client.start_workflow(
                adapter.MapQueriesWorkflow.run,
                adapter.MapQueriesInput(
                    queries=[
                        adapter.query_activity_input(
                            session,
                            query_id="q",
                            tail="Question",
                            output_name="FeedbackAnswer",
                            prefix_ref=prefix_ref,
                            ref_namespace="synthetic" if factory else None,
                        )
                    ],
                    batch_size=1 if batch else None,
                    on_query_failure="return",
                ),
                id=f"feedback-{uuid4()}",
                task_queue=queue,
                execution_timeout=timedelta(seconds=30),
            )
            result = (await handle.result())[0]
            assert result.failure_kind is None
            assert result.feedback_retries_used == 1
            assert result.meta["feedback_retries_used"] == 1
            assert attempts == [1, 1]
            encoded = await store.get(result.output_ref) if result.output_ref else result.output_json
            assert json.loads(encoded) == {"text": "repaired"}
            first, second = scripted.calls
            assert first[0][0].parts[0].content == second[0][0].parts[0].content == session.rendered_documents
            assert "synthetic-validator-feedback" in feedback_text(second[0])
            history = await handle.fetch_history()
            children = [
                await env.client.get_workflow_handle(
                    event.start_child_workflow_execution_initiated_event_attributes.workflow_id
                ).fetch_history()
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
            await replayer.replay_workflow(child)


@pytest.mark.parametrize("plugin", [False, True])
async def test_exhausted_feedback_failure_keeps_only_safe_count(plugin: bool) -> None:
    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        return tool_result(info, "FeedbackAnswer", {"invalid-field": "synthetic-private-value"})

    scripted = Scripted(handler)
    session = build(
        scripted.model(),
        documents="Synthetic evidence.",
        instructions="Answer.",
        outputs=[FeedbackAnswer],
        policy=Policy(retries=0, output_feedback_retries=1),
        warmup=False,
    )
    adapter.register_session(session)
    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
        queue = f"feedback-exhausted-{uuid4()}"
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[adapter.MapQueriesWorkflow],
            activities=[adapter.run_query_activity],
            workflow_runner=runner,
        ):
            handle = await env.client.start_workflow(
                adapter.MapQueriesWorkflow.run,
                adapter.MapQueriesInput(
                    queries=[
                        adapter.query_activity_input(
                            session,
                            query_id="q",
                            tail="Question",
                            output_name="FeedbackAnswer",
                        )
                    ],
                    on_query_failure="return",
                ),
                id=f"feedback-exhausted-{uuid4()}",
                task_queue=queue,
                execution_timeout=timedelta(seconds=30),
            )
            result = (await handle.result())[0]
            assert result.failure_kind == "retryable_exhausted" and result.failure_type == "validation"
            assert result.feedback_retries_used == 1 and result.meta == {"feedback_retries_used": 1}
            assert len(scripted.calls) == 2
            assert result.output_json is None and result.messages_json is None
            assert "synthetic-private-value" not in result.failure_message
            history = await handle.fetch_history()
        await Replayer(workflows=[adapter.MapQueriesWorkflow], plugins=plugins, workflow_runner=runner).replay_workflow(
            history
        )
