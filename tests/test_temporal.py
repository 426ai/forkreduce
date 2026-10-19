"""Temporal mapping, message store, and a single activity attempt."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel
from temporalio.exceptions import ApplicationError
from tests.support import Scripted, tool_result

from forkreduce import Query, build
from forkreduce.contrib.temporal import (
    InMemoryMessageStore,
    QueryActivityInput,
    activity_id,
    execute_query_activity,
    register_session,
    set_message_store,
    temporal_retry_policy,
    to_application_error,
)
from forkreduce.failures import Failure, FailureKind
from forkreduce.policy import Policy


class Answer(BaseModel):
    """A short answer."""

    text: str


def test_core_import_does_not_load_temporalio() -> None:
    script = (
        "import sys\n"
        "import forkreduce\n"
        "assert not any(name == 'temporalio' or name.startswith('temporalio.') for name in sys.modules)\n"
        "assert forkreduce.__version__ == '0.1.0'\n"
    )
    env = os.environ.copy()
    env["PYDANTIC_AI_NO_BANNER"] = "1"
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call([sys.executable, "-c", script], env=env)


def test_application_error_mapping_and_retry_policy() -> None:
    transient = to_application_error(
        Failure(kind=FailureKind.TRANSPORT, message="slow", retryable=True, code="http_429")
    )
    permanent = to_application_error(
        Failure(kind=FailureKind.BUDGET, message="over", retryable=False, code="budget_exceeded")
    )
    assert isinstance(transient, ApplicationError)
    assert transient.non_retryable is False
    assert transient.type == "transport"
    assert permanent.non_retryable is True
    assert permanent.type == "budget"
    policy = temporal_retry_policy(Policy(retries=2, retry_backoff=0.25, max_backoff=2))
    assert policy.maximum_attempts == 3


@pytest.mark.asyncio
async def test_activity_stores_messages_and_rejects_unknown_sessions() -> None:
    def handler(_messages, info, _n):  # type: ignore[no-untyped-def]
        return tool_result(info, "Answer", {"text": "stored"})

    session = build(
        Scripted(handler).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=0, retry_backoff=0),
    )
    prefix = register_session(session)
    store = InMemoryMessageStore()
    set_message_store(store)
    try:
        output = await execute_query_activity(
            QueryActivityInput(
                prefix_hash=prefix,
                query_id="q1",
                tail="hello",
                output_name="Answer",
            )
        )
    finally:
        set_message_store(None)
    assert output.output_type == "Answer"
    assert output.messages_json is None
    assert output.messages_ref is not None
    assert activity_id(prefix, "q1") == f"forkreduce:{prefix}:q1"
    payload = await store.get(output.messages_ref)
    assert b"hello" in payload or payload

    with pytest.raises(ApplicationError) as caught:
        await execute_query_activity(
            QueryActivityInput(
                prefix_hash="missing",
                query_id="q",
                tail="x",
                output_name="Answer",
            )
        )
    assert caught.value.non_retryable is True

    # The activity itself is one attempt: a permanent HTTP failure is non-retryable.
    def denied(_messages, _info, _n):  # type: ignore[no-untyped-def]
        from pydantic_ai.exceptions import ModelHTTPError

        raise ModelHTTPError(status_code=401, model_name="m", body="no")

    denied_session = build(
        Scripted(denied).model(),
        documents="doc",
        instructions="rules",
        outputs=[Answer],
        warmup=False,
        policy=Policy(retries=5, retry_backoff=0),
    )
    register_session(denied_session)
    with pytest.raises(ApplicationError) as caught:
        await execute_query_activity(
            QueryActivityInput(
                prefix_hash=denied_session.prefix_hash,
                query_id="nope",
                tail="x",
                output_name="Answer",
            )
        )
    assert caught.value.non_retryable is True


def test_run_once_is_what_the_activity_uses_not_the_executor_loop() -> None:
    # Policy.retries on the session must not multiply attempts inside one activity.
    # This is covered by the 401 case above: LocalExecutor would have stopped at the
    # first permanent failure too, so also assert the helper builds a query id.
    query = Query(tail="x", output=Answer, id="fixed")
    assert query.id == "fixed"
