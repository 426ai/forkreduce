"""Complete offline Temporal integration with synthetic data and local references.

Set TEMPORAL_TEST_SERVER to an existing cached Temporal test-server binary.
Run from this checkout with PYTHONPATH=src:. and add --plugin for PydanticAIPlugin.
No test-server download, provider request, tracing export or credential is used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from pydantic_ai import models  # noqa: E402
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin  # noqa: E402
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.models.instrumented import InstrumentationSettings  # noqa: E402
from pydantic_ai.usage import RequestUsage  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions  # noqa: E402

from forkreduce import ActivityOptions, OutputSpec, Policy, Session, build  # noqa: E402
from forkreduce.contrib.temporal import (  # noqa: E402
    AfterQueryDecision,
    AfterQueryInput,
    FrozenPrefix,
    MapQueriesInput,
    QueryActivityOutput,
    StoreContext,
    delete_ref,
    forkreduce_temporal_names,
    freeze_prefix,
    query_activity_input,
    set_default_session_factory,
    set_message_store,
    set_prefix_store,
)

models.ALLOW_MODEL_REQUESTS = False


class DurableStore:
    """Local SQLite demonstration, not a distributed or authorization backend."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS blobs (ref TEXT PRIMARY KEY, body BLOB)")

    def execute(self, sql: str, args: tuple[Any, ...]) -> Any:
        with sqlite3.connect(self.path) as db:
            return db.execute(sql, args).fetchone()

    @staticmethod
    def routed(ref: str, context: StoreContext) -> None:
        if not context.namespace or not ref.startswith(quote(context.namespace, safe="") + "/"):
            raise ValueError("Reference reads require the original routing namespace")

    async def put(self, key: str, payload: bytes, *, context: StoreContext) -> str:
        self.routed(key, context)
        ref = f"{key}:{hashlib.sha256(payload).hexdigest()}"
        await asyncio.to_thread(self.execute, "INSERT OR IGNORE INTO blobs VALUES (?, ?)", (ref, payload))
        return ref

    async def get(self, ref: str, *, context: StoreContext) -> bytes:
        self.routed(ref, context)
        row = await asyncio.to_thread(self.execute, "SELECT body FROM blobs WHERE ref = ?", (ref,))
        if row is None:
            raise KeyError(ref)
        return bytes(row[0])

    async def delete(self, ref: str, *, context: StoreContext) -> None:
        self.routed(ref, context)
        await asyncio.to_thread(self.execute, "DELETE FROM blobs WHERE ref = ?", (ref,))


async def main(*, plugin: bool = False) -> None:
    binary = os.environ.get("TEMPORAL_TEST_SERVER")
    if not binary or not Path(binary).is_file():
        raise SystemExit("Set TEMPORAL_TEST_SERVER to a cached test-server binary; no download is attempted.")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentation = InstrumentationSettings(tracer_provider=provider, include_content=False)
    request_counts: Counter[str] = Counter()
    review_events: list[tuple[str, int, int]] = []
    raw_document = "Synthetic evidence: a label and a count.\x00\n"

    async def reply(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        query_id = activity.info().activity_id.rsplit(":", 1)[-1]
        request_counts[query_id] += 1
        if query_id == "label":
            value: str | int = 17 if request_counts[query_id] == 1 else "draft"
        else:
            value = "repaired" if query_id == "label#r1" else 7
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"value": value})],
            usage=RequestUsage(input_tokens=100, output_tokens=10, cache_read_tokens=64),
        )

    def prepare() -> Session:
        return build(
            FunctionModel(reply),
            documents=raw_document,
            instructions="Use only the synthetic evidence.",
            outputs=[],
            dynamic_outputs="generic_tool",
            document_format="raw",
            message_layout="single",
            policy=Policy(retries=0, output_feedback_retries=1),
            warmup=False,
        )

    def restore(frozen: FrozenPrefix) -> Session:
        assert frozen.app_meta["schema_version"] == 1
        return build(
            FunctionModel(reply),
            documents=frozen.documents,
            instructions=frozen.instructions,
            outputs=[],
            output_mode=frozen.output_mode,
            dynamic_outputs=frozen.dynamic_outputs,
            document_format=frozen.document_format,
            message_layout=frozen.message_layout,
            on_unforced_tools=frozen.on_unforced_tools,
            policy=Policy(retries=0, output_feedback_retries=frozen.output_feedback_retries),
            instrument=instrumentation,
            warmup=False,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="forkreduce-offline-") as directory:
            store = DurableStore(Path(directory) / "references.sqlite")
            session = prepare()
            assert session.rendered_documents == raw_document
            namespace = "offline-demo"
            prefix_ref = await freeze_prefix(
                session,
                store,
                app_meta={"ref_namespace": namespace, "schema_version": 1},
            )
            # No registered sessions/exact factories are needed after this point.
            set_default_session_factory(restore)
            set_prefix_store(store)
            set_message_store(store)
            label = OutputSpec(
                "Label",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            )
            count = OutputSpec(
                "Count",
                {
                    "type": "object",
                    "properties": {"value": {"type": "integer", "minimum": 0}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            )

            @activity.defn(name="demo.review")
            async def review(data: AfterQueryInput) -> AfterQueryDecision:
                review_events.append((data.query.query_id, data.round, data.output.feedback_retries_used))
                if data.output.failure_kind is not None:
                    return AfterQueryDecision("done", result_meta={"status": "failed"})
                assert data.output.output_ref is not None
                context = StoreContext(kind="output", prefix_hash=data.query.prefix_hash, namespace=namespace)
                output = json.loads(await store.get(data.output.output_ref, context=context))
                if output["value"] == "draft":
                    return AfterQueryDecision("follow_up", follow_up_tail="Revise the draft using the evidence.")
                return AfterQueryDecision("done", result_meta={"status": "accepted", "round": data.round})

            names = forkreduce_temporal_names(prefix="demo.")
            map_workflow, batch_workflow = names.workflows
            runner = SandboxedWorkflowRunner(
                restrictions=SandboxRestrictions.default.with_passthrough_modules("forkreduce", "pydantic_ai")
            )
            plugins = [PydanticAIPlugin()] if plugin else []
            async with await WorkflowEnvironment.start_time_skipping(
                plugins=plugins,
                test_server_existing_path=binary,
            ) as env:
                parent_queue, model_queue = f"parent-{uuid4()}", f"models-{uuid4()}"
                async with (
                    Worker(env.client, task_queue=parent_queue, workflows=[map_workflow], workflow_runner=runner),
                    Worker(
                        env.client,
                        task_queue=model_queue,
                        workflows=[batch_workflow],
                        activities=[*names.activities, review],
                        workflow_runner=runner,
                    ),
                ):
                    handle = await env.client.start_workflow(
                        names.map_workflow_name,
                        MapQueriesInput(
                            queries=[
                                query_activity_input(
                                    session,
                                    query_id=query_id,
                                    tail="Return the requested value.",
                                    output_spec=spec,
                                    prefix_ref=prefix_ref,
                                    ref_namespace=namespace,
                                )
                                for query_id, spec in (("label", label), ("count", count))
                            ],
                            batch_size=2,
                            child_concurrency=2,
                            activity_concurrency=2,
                            task_queue=model_queue,
                            warmup="first",
                            on_query_failure="return",
                            after_query_activity="demo.review",
                            after_query_options=ActivityOptions(start_to_close=10, maximum_attempts=1),
                            max_rounds=1,
                            activity_options=ActivityOptions(
                                start_to_close=60, heartbeat_timeout=10, maximum_attempts=2
                            ),
                        ),
                        id=f"offline-map-{uuid4()}",
                        task_queue=parent_queue,
                        result_type=list[QueryActivityOutput],
                        execution_timeout=timedelta(seconds=60),
                    )
                    results = await handle.result()
                    assert all(result.failure_kind is None for result in results)
                    assert [result.query_id for result in results] == ["label#r1", "count"]
                    history = await handle.fetch_history()
                    child_histories = [
                        await env.client.get_workflow_handle(
                            event.start_child_workflow_execution_initiated_event_attributes.workflow_id
                        ).fetch_history()
                        for event in history.events
                        if event.HasField("start_child_workflow_execution_initiated_event_attributes")
                    ]
                replayer = Replayer(workflows=list(names.workflows), plugins=plugins, workflow_runner=runner)
                for recorded in [history, *child_histories]:
                    await replayer.replay_workflow(recorded)
                values = []
                for result in results:
                    assert result.output_ref is not None
                    context = StoreContext(kind="output", prefix_hash=result.prefix_hash, namespace=namespace)
                    values.append(json.loads(await store.get(result.output_ref, context=context)))
                    # The application, not the library, chooses the retention boundary.
                    await delete_ref(store, result.output_ref, context=context)
                    await delete_ref(store, result.output_ref, context=context)
                chats = [
                    span
                    for span in exporter.get_finished_spans()
                    if span.attributes.get("gen_ai.operation.name") == "chat"
                ]
                assert values == [{"value": "repaired"}, {"value": 7}]
                assert request_counts == {"label": 2, "label#r1": 1, "count": 1}
                assert review_events == [("label", 0, 1), ("label#r1", 1, 0), ("count", 0, 0)]
                assert len(chats) == sum(request_counts.values())
                print(
                    json.dumps(
                        {
                            "converter": "plugin" if plugin else "default",
                            "prefix": session.prefix_hash[:12],
                            "outputs": values,
                            "model_requests": sum(request_counts.values()),
                            "trace_model_spans": len(chats),
                            "replayed_histories": 1 + len(child_histories),
                            "result_meta": [result.result_meta for result in results],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
    finally:
        set_default_session_factory(None)
        set_prefix_store(None)
        set_message_store(None)
        provider.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", action="store_true", help="Use the official PydanticAIPlugin converter")
    asyncio.run(main(plugin=parser.parse_args().plugin))
