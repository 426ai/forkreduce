"""Safe terminal query records through real workers and deterministic replay."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from temporalio import activity, workflow
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    CancelledError,
    RetryState,
    TimeoutError,
    TimeoutType,
)
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from pydantic_ai.exceptions import ModelHTTPError
    from temporalio.client import WorkflowFailureError
    from tests.support import Scripted, tool_result
    from tests.test_temporal_runtime import sandbox_runner, server_options

    from forkreduce import ActivityOptions, build
    from forkreduce.contrib import temporal as adapter
    from forkreduce.failures import Failure, FailureKind
    from forkreduce.policy import Policy


class FailureAnswer(BaseModel):
    text: str


def activity_error(cause: BaseException, retry_state: RetryState) -> ActivityError:
    error = ActivityError(
        "untrusted activity wrapper text",
        scheduled_event_id=1,
        started_event_id=2,
        identity="offline-worker",
        activity_type="forkreduce.run_query",
        activity_id="query",
        retry_state=retry_state,
    )
    error.__cause__ = cause
    return error


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("batch_size", [None, 3])
async def test_return_mixed_query_failures_and_replay(plugin: bool, batch_size: int | None) -> None:
    sensitive = "synthetic-private-payload-marker"
    seen: dict[str, list[int]] = {name: [] for name in ("success", "permanent", "exhausted")}

    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        current = activity.info()
        query_id = current.activity_id.rsplit(":", 1)[1]
        seen[query_id].append(current.attempt)
        if query_id == "permanent":
            raise ModelHTTPError(401, sensitive, body={"private": sensitive})
        if query_id == "exhausted" or current.attempt == 1:
            raise ModelHTTPError(503, sensitive, body={"private": sensitive})
        return tool_result(info, "FailureAnswer", {"text": "successful answer"})

    class RecordingStore(adapter.InMemoryMessageStore):
        async def put(self, key: str, payload: bytes) -> str:
            assert activity.in_activity()
            return await super().put(key, payload)

    session = build(
        Scripted(handler).model(),
        documents=f"Synthetic evidence {sensitive}",
        instructions="Use the evidence.",
        outputs=[FailureAnswer],
        policy=Policy(retries=3, retry_backoff=0.001, max_backoff=0.001),
    )
    prefix = adapter.register_session(session)
    store = RecordingStore()
    adapter.set_message_store(store if batch_size else None)
    plugins = [PydanticAIPlugin()] if plugin else []
    runner = sandbox_runner()
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"failure-return-{uuid4()}"
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
                                query_id=name,
                                tail=f"{name} {sensitive}",
                                output_name="FailureAnswer",
                            )
                            for name in seen
                        ],
                        batch_size=batch_size,
                        on_query_failure="return",
                    ),
                    id=f"failure-return-{uuid4()}",
                    task_queue=queue,
                    execution_timeout=timedelta(seconds=30),
                )
                results = await asyncio.wait_for(handle.result(), 30)
                assert [result.query_id for result in results] == list(seen)
                assert [result.attempts for result in results if result.failure_kind is not None] == [1, 4]
                assert [result.failure_kind for result in results] == [None, "permanent", "retryable_exhausted"]
                assert seen == {"success": [1, 2], "permanent": [1], "exhausted": [1, 2, 3, 4]}
                success, permanent, exhausted = results
                assert success.failure_message is None
                if batch_size:
                    assert success.output_json is None and success.output_ref
                    assert success.messages_json is None and success.messages_ref
                    assert json.loads(await store.get(success.output_ref)) == {"text": "successful answer"}
                    assert len(store._data) == 2
                else:
                    assert json.loads(success.output_json or "null") == {"text": "successful answer"}
                    assert success.messages_json and success.messages_ref is None
                for result in (permanent, exhausted):
                    assert result.cache_hit_tokens == 0
                    assert result.output_json is None and result.output_type is None and result.output_ref is None
                    assert result.messages_json is None and result.messages_ref is None
                    encoded = json.dumps(asdict(result))
                    assert sensitive not in encoded
                    assert "ModelHTTPError" not in encoded
                    assert result.failure_message == adapter._FAILURE_MESSAGES[result.failure_kind]
                history = await handle.fetch_history()
                children = [
                    event.start_child_workflow_execution_initiated_event_attributes.workflow_id
                    for event in history.events
                    if event.HasField("start_child_workflow_execution_initiated_event_attributes")
                ]
                assert len(children) == (1 if batch_size else 0)
                child_histories = [await env.client.get_workflow_handle(child).fetch_history() for child in children]
            replayer = Replayer(
                workflows=[adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow],
                plugins=plugins,
                workflow_runner=runner,
            )
            await replayer.replay_workflow(history)
            for child_history in child_histories:
                await replayer.replay_workflow(child_history)
    finally:
        adapter.unregister_session(prefix)
        adapter.set_message_store(None)


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("input_mode", ["legacy", "default", "raise"])
async def test_raise_mode_keeps_terminal_failure_behavior(plugin: bool, input_mode: str) -> None:
    attempts: list[int] = []

    def handler(_messages, _info, _n):  # type: ignore[no-untyped-def]
        attempts.append(activity.info().attempt)
        raise ModelHTTPError(401, "offline", body={"error": "synthetic provider error"})

    session = build(
        Scripted(handler).model(),
        documents="Raise mode evidence.",
        instructions="Answer.",
        outputs=[FailureAnswer],
        policy=Policy(retries=3),
    )
    prefix = adapter.register_session(session)
    adapter.set_message_store(adapter.InMemoryMessageStore())
    plugins = [PydanticAIPlugin()] if plugin else []
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"failure-raise-{uuid4()}"
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=[adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow],
                activities=[adapter.run_query_activity],
                workflow_runner=sandbox_runner(),
            ):
                queries = [
                    adapter.query_activity_input(
                        session, query_id="permanent", tail="Question", output_name="FailureAnswer"
                    )
                ]
                payload = (
                    queries
                    if input_mode == "legacy"
                    else adapter.MapQueriesInput(queries=queries, batch_size=1)
                    if input_mode == "default"
                    else adapter.MapQueriesInput(queries=queries, on_query_failure="raise")
                )
                with pytest.raises(WorkflowFailureError) as caught:
                    await env.client.execute_workflow(
                        adapter.MapQueriesWorkflow.run,
                        payload,
                        id=f"failure-raise-{uuid4()}",
                        task_queue=queue,
                        execution_timeout=timedelta(seconds=30),
                    )
                cause: BaseException | None = caught.value
                while cause is not None and not isinstance(cause, ApplicationError):
                    cause = cause.__cause__
                assert isinstance(cause, ApplicationError) and cause.type == "permanent"
                assert cause.details[0] == "permanent"
                assert cause.details[1] == {"forkreduce_attempt": 1}
                assert attempts == [1]
    finally:
        adapter.unregister_session(prefix)
        adapter.set_message_store(None)


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("batch_size", [None, 1])
async def test_return_mode_does_not_swallow_workflow_cancellation(plugin: bool, batch_size: int | None) -> None:
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def handler(_messages, _info, _n):  # type: ignore[no-untyped-def]
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    session = build(
        Scripted(handler).model(),
        documents="Cancellation evidence.",
        instructions="Answer.",
        outputs=[FailureAnswer],
        policy=Policy(retries=0),
        activity_options=ActivityOptions(start_to_close=20, heartbeat_timeout=0.2),
    )
    prefix = adapter.register_session(session)
    adapter.set_message_store(adapter.InMemoryMessageStore())
    plugins = [PydanticAIPlugin()] if plugin else []
    try:
        async with await WorkflowEnvironment.start_time_skipping(plugins=plugins, **server_options()) as env:
            queue = f"failure-cancel-{uuid4()}"
            async with Worker(
                env.client,
                task_queue=queue,
                workflows=[adapter.MapQueriesWorkflow, adapter.QueryBatchWorkflow],
                activities=[adapter.run_query_activity],
                workflow_runner=sandbox_runner(),
                max_heartbeat_throttle_interval=timedelta(milliseconds=20),
                default_heartbeat_throttle_interval=timedelta(milliseconds=20),
            ):
                handle = await env.client.start_workflow(
                    adapter.MapQueriesWorkflow.run,
                    adapter.MapQueriesInput(
                        queries=[
                            adapter.query_activity_input(
                                session, query_id="cancel", tail="Question", output_name="FailureAnswer"
                            )
                        ],
                        batch_size=batch_size,
                        on_query_failure="return",
                    ),
                    id=f"failure-cancel-{uuid4()}",
                    task_queue=queue,
                )
                await asyncio.wait_for(entered.wait(), 10)
                await handle.cancel()
                with pytest.raises(WorkflowFailureError) as caught:
                    await asyncio.wait_for(handle.result(), 10)
                assert isinstance(caught.value.cause, CancelledError)
                if batch_size is None:
                    await asyncio.wait_for(stopped.wait(), 10)
    finally:
        adapter.unregister_session(prefix)
        adapter.set_message_store(None)


@pytest.mark.parametrize(
    ("cause", "state", "kind", "attempts"),
    [
        (
            ApplicationError("private", "permanent", {"forkreduce_attempt": 2}, type="permanent", non_retryable=True),
            RetryState.NON_RETRYABLE_FAILURE,
            "permanent",
            2,
        ),
        (
            ApplicationError("private", "budget", {"forkreduce_attempt": 1}, type="budget", non_retryable=True),
            RetryState.NON_RETRYABLE_FAILURE,
            "budget",
            1,
        ),
        (
            ApplicationError("private", "transport", {"forkreduce_attempt": 3}, type="transport"),
            RetryState.MAXIMUM_ATTEMPTS_REACHED,
            "retryable_exhausted",
            3,
        ),
        (
            ApplicationError("private", "validation", {"forkreduce_attempt": 2}, type="validation"),
            RetryState.MAXIMUM_ATTEMPTS_REACHED,
            "retryable_exhausted",
            2,
        ),
        (
            ApplicationError("private", type="unexpected", non_retryable=True),
            RetryState.NON_RETRYABLE_FAILURE,
            "permanent",
            0,
        ),
        (
            TimeoutError(
                "private", type=TimeoutType.START_TO_CLOSE, last_heartbeat_details=[{"forkreduce_attempt": 2}]
            ),
            RetryState.MAXIMUM_ATTEMPTS_REACHED,
            "timeout",
            2,
        ),
        (
            TimeoutError("private", type=TimeoutType.SCHEDULE_TO_CLOSE, last_heartbeat_details=[]),
            RetryState.TIMEOUT,
            "timeout",
            0,
        ),
        (CancelledError("private"), RetryState.CANCEL_REQUESTED, "cancelled", 0),
        (RuntimeError("private"), RetryState.INTERNAL_SERVER_ERROR, "activity", 0),
    ],
)
async def test_terminal_error_mapping_never_exposes_error_text(
    monkeypatch, cause: BaseException, state: RetryState, kind: str, attempts: int
) -> None:  # type: ignore[no-untyped-def]
    query = adapter.QueryActivityInput("prefix", "q", "private query", "FailureAnswer", max_attempts=99)
    error = activity_error(cause, state)

    async def execute(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise error

    monkeypatch.setattr(adapter.workflow, "execute_activity", execute)
    results = await adapter.MapQueriesWorkflow().run(
        adapter.MapQueriesInput(queries=[query], on_query_failure="return")
    )
    result = results[0]
    assert result.failure_kind == kind
    assert result.attempts == attempts
    assert "private" not in json.dumps(asdict(result))
    assert "untrusted" not in result.failure_message
    assert len(result.failure_message) < 100
    assert result.cache_hit_tokens == 0
    assert result.output_json is None and result.output_ref is None and result.output_type is None
    assert result.messages_json is None and result.messages_ref is None


@pytest.mark.parametrize("wrapped", [False, True])
async def test_requested_cancellation_propagates_even_when_wrapped(monkeypatch, wrapped: bool) -> None:  # type: ignore[no-untyped-def]
    cause = CancelledError("cancelled") if wrapped else asyncio.CancelledError()
    error = activity_error(cause, RetryState.CANCEL_REQUESTED) if wrapped else cause
    query = adapter.QueryActivityInput("prefix", "q", "question", "FailureAnswer")

    async def execute(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise error

    monkeypatch.setattr(adapter.workflow, "execute_activity", execute)
    monkeypatch.setattr(adapter.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(adapter.workflow, "info", lambda: SimpleNamespace(parent=None))
    monkeypatch.setattr(adapter.workflow, "cancellation_reason", lambda: "")
    with pytest.raises(type(error)) as caught:
        await adapter.MapQueriesWorkflow().run(adapter.MapQueriesInput(queries=[query], on_query_failure="return"))
    assert caught.value is error


async def test_batch_exempts_only_explicit_failure_records(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    query = adapter.QueryActivityInput("prefix", "q", "question", "FailureAnswer")
    result = adapter.QueryActivityOutput("q", "prefix", None, None, 0, 1)

    async def execute(*args, **kwargs):  # type: ignore[no-untyped-def]
        return result

    monkeypatch.setattr(adapter.workflow, "execute_activity", execute)
    with pytest.raises(ApplicationError, match="references"):
        await adapter.QueryBatchWorkflow().run(adapter.MapQueriesInput(queries=[query], on_query_failure="return"))
    result.failure_kind = "permanent"
    result.failure_message = adapter._FAILURE_MESSAGES["permanent"]
    assert await adapter.QueryBatchWorkflow().run(
        adapter.MapQueriesInput(queries=[query], on_query_failure="return")
    ) == [result]


def test_failure_details_preserve_kind_and_add_observed_attempt(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(adapter.activity, "in_activity", lambda: True)
    monkeypatch.setattr(adapter.activity, "info", lambda: SimpleNamespace(attempt=3))
    error = adapter.to_application_error(Failure(FailureKind.TRANSPORT, "private", True))
    assert error.details == ("transport", {"forkreduce_attempt": 3})


async def test_activity_payload_carries_parent_workflow_identity(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured = []
    query = adapter.QueryActivityInput("prefix", "q", "question", "FailureAnswer")

    async def execute(_activity, payload, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(payload)
        return adapter.QueryActivityOutput("q", "prefix", "{}", "FailureAnswer", 0, 1)

    monkeypatch.setattr(adapter.workflow, "execute_activity", execute)
    monkeypatch.setattr(adapter.workflow, "in_workflow", lambda: True)
    monkeypatch.setattr(adapter.workflow, "info", lambda: SimpleNamespace(parent=SimpleNamespace(workflow_id="parent")))
    await adapter._execute_activity(query, adapter.activity_retry_policy(query))
    assert captured[0].parent_workflow_id == "parent"
    monkeypatch.setattr(adapter.workflow, "info", lambda: SimpleNamespace(parent=None))
    await adapter._execute_activity(captured[0], adapter.activity_retry_policy(query))
    assert captured[1].parent_workflow_id is None
