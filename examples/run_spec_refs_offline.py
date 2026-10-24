"""Generate bounded batches in an activity; never send a large query/spec list.

Requires checkout dev/Temporal extras and an existing TEMPORAL_TEST_SERVER binary.
Run with PYTHONPATH=src:. python examples/run_spec_refs_offline.py [--plugin].
Only synthetic models/data; no downloads, provider requests or telemetry exports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import models
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from examples.run_temporal_offline import DurableStore
    from forkreduce import ActivityOptions, OutputSpec, Policy, build
    from forkreduce.contrib.temporal import (
        OutputSpecWriter,
        QueryActivityInput,
        QueryActivityOutput,
        StoreContext,
        execute_query,
        freeze_prefix,
        plan_batches,
        query_activity_input,
        run_query_activity,
        set_default_session_factory,
        set_message_store,
        set_prefix_store,
    )


@dataclass(frozen=True)
class GeneratedQueriesInput:
    prefix_hash: str
    prefix_ref: str
    namespace: str
    count: int = 300
    batch_size: int = 20


@dataclass(frozen=True)
class GenerateBatchInput:
    job: GeneratedQueriesInput
    indices: list[int]


@workflow.defn(name="example.generated_queries")
class GeneratedQueriesWorkflow:
    @workflow.run
    async def run(self, job: GeneratedQueriesInput) -> list[QueryActivityOutput]:
        outputs = []
        for indices in plan_batches(list(range(job.count)), job.batch_size):
            queries = await workflow.execute_activity(
                "example.generate_queries",
                GenerateBatchInput(job, indices),
                result_type=list[QueryActivityInput],
                start_to_close_timeout=timedelta(seconds=60),
            )
            # A bounded batch limits both activity count and preparation-result size.
            outputs.extend(
                await asyncio.gather(
                    *(
                        execute_query(
                            query, ActivityOptions(start_to_close=60, maximum_attempts=1), require_references=True
                        )
                        for query in queries
                    )
                )
            )
        return outputs


async def run(*, plugin: bool = False) -> dict[str, int | bool]:
    binary = os.environ.get("TEMPORAL_TEST_SERVER")
    if not binary or not Path(binary).is_file():
        raise RuntimeError("Set TEMPORAL_TEST_SERVER to an existing cached binary; downloads are disabled")
    old_allow = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    requests: Counter[str] = Counter()

    def reply(_messages, info):
        query_id = activity.info().activity_id.rsplit(":", 1)[-1]
        index = int(query_id[1:])
        requests[query_id] += 1
        value = "invalid" if index == 0 and requests[query_id] == 1 else index
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"value": value})])

    def restore(frozen=None):
        return build(
            FunctionModel(reply),
            documents="Synthetic shared evidence." if frozen is None else frozen.documents,
            instructions="Return the requested object.",
            outputs=[],
            dynamic_outputs="generic_tool",
            policy=Policy(retries=0, output_feedback_retries=1),
            warmup=False,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="forkreduce-spec-refs-") as directory:
            store = DurableStore(Path(directory) / "refs.sqlite")
            session = restore()
            writer = OutputSpecWriter(store)
            namespace = "generated-example"
            prefix_ref = await freeze_prefix(session, store, app_meta={"ref_namespace": namespace})
            set_default_session_factory(restore)
            set_prefix_store(store)
            set_message_store(store)

            @activity.defn(name="example.generate_queries")
            async def generate(data: GenerateBatchInput) -> list[QueryActivityInput]:
                queries = []
                for index in data.indices:
                    spec = OutputSpec(
                        "Value",
                        {
                            "type": "object",
                            "description": "Synthetic schema padding. " + "x" * 11_500,
                            "properties": {"value": {"type": "integer", "const": index}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    )
                    ref = await writer.store(spec, ref_namespace=data.job.namespace)
                    queries.append(
                        query_activity_input(
                            session,
                            query_id=f"q{index}",
                            tail=f"Return value {index}.",
                            output_spec=ref,
                            prefix_ref=data.job.prefix_ref,
                            ref_namespace=data.job.namespace,
                        )
                    )
                return queries

            runner = SandboxedWorkflowRunner(
                restrictions=SandboxRestrictions.default.with_passthrough_modules(
                    "forkreduce",
                    "pydantic_ai",
                    "examples.run_spec_refs_offline",
                )
            )
            plugins = [PydanticAIPlugin()] if plugin else []
            async with await WorkflowEnvironment.start_time_skipping(
                test_server_existing_path=binary,
                plugins=plugins,
            ) as env:
                queue = f"spec-refs-{uuid4()}"
                async with Worker(
                    env.client,
                    task_queue=queue,
                    workflows=[GeneratedQueriesWorkflow],
                    activities=[generate, run_query_activity],
                    workflow_runner=runner,
                ):
                    handle = await env.client.start_workflow(
                        GeneratedQueriesWorkflow.run,
                        GeneratedQueriesInput(session.prefix_hash, prefix_ref, namespace),
                        id=f"generated-{uuid4()}",
                        task_queue=queue,
                        execution_timeout=timedelta(seconds=120),
                    )
                    outputs = await handle.result()
                    history = await handle.fetch_history()
                await Replayer(
                    workflows=[GeneratedQueriesWorkflow], plugins=plugins, workflow_runner=runner
                ).replay_workflow(history)
                activity_inputs, activity_results, workflow_inputs = [], [], []
                for event in history.events:
                    if event.HasField("activity_task_scheduled_event_attributes"):
                        activity_inputs.extend(
                            p.ByteSize() for p in event.activity_task_scheduled_event_attributes.input.payloads
                        )
                    if event.HasField("activity_task_completed_event_attributes"):
                        activity_results.extend(
                            p.ByteSize() for p in event.activity_task_completed_event_attributes.result.payloads
                        )
                    if event.HasField("workflow_execution_started_event_attributes"):
                        workflow_inputs.extend(
                            p.ByteSize() for p in event.workflow_execution_started_event_attributes.input.payloads
                        )
                context = StoreContext("output", session.prefix_hash, namespace=namespace)
                values = [json.loads(await store.get(output.output_ref, context=context)) for output in outputs]
                assert values == [{"value": i} for i in range(300)]
                assert sum(requests.values()) == 301
                assert outputs[0].feedback_retries_used == 1
                assert max([*activity_inputs, *activity_results, *workflow_inputs]) < 512 * 1024
                return {
                    "plugin": plugin,
                    "queries": len(outputs),
                    "model_requests": sum(requests.values()),
                    "workflow_input_max_bytes": max(workflow_inputs),
                    "activity_input_max_bytes": max(activity_inputs),
                    "activity_result_max_bytes": max(activity_results),
                    "replayed_histories": 1,
                }
    finally:
        models.ALLOW_MODEL_REQUESTS = old_allow
        set_default_session_factory(None)
        set_prefix_store(None)
        set_message_store(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", action="store_true")
    print(json.dumps(asyncio.run(run(plugin=parser.parse_args().plugin)), indent=2))
