"""Explicit fresh follow-ups retain prefixes while controlling history and schemas."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    from tests.support import Scripted, tool_result, user_text
    from tests.test_temporal_hooks import running_map
    from tests.test_temporal_output_specs import output_specs

    from forkreduce import Policy, build
    from forkreduce.contrib import temporal as adapter


@pytest.fixture(autouse=True)
def isolated_registrations(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_SESSIONS", "_FINGERPRINTS", "_FACTORIES"):
        monkeypatch.setattr(adapter, name, {})
    for name in ("_DEFAULT_FACTORY", "_STORE", "_PREFIX_STORE"):
        monkeypatch.setattr(adapter, name, None)


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize(
    ("history", "switch_schema"),
    [
        ("legacy", False),
        ("continue", False),
        ("continue", True),
        ("explicit", True),
        ("fresh", False),
        ("fresh", True),
        ("tail", False),
    ],
)
async def test_follow_up_history_schema_selection_and_replay(plugin: bool, history: str, switch_schema: bool) -> None:
    requests = []
    activity_ids = []
    hooks = []
    explicit_ref = None
    specs = output_specs()

    def reply(messages, info, number):  # type: ignore[no-untyped-def]
        requests.append(copy.deepcopy(messages))
        activity_ids.append(activity.info().activity_id)
        value = 7 if number > 1 and switch_schema else "OK"
        return tool_result(info, info.output_tools[0].name, {"value": value})

    session = build(
        Scripted(reply).model(),
        outputs=[],
        documents="Synthetic fresh-history evidence.",
        instructions="Answer from the evidence.",
        dynamic_outputs="generic_tool",
        output_mode="tools",
        policy=Policy(retries=0),
        warmup=False,
    )
    adapter.register_session(session)
    store = adapter.InMemoryMessageStore()
    adapter.set_message_store(store)
    query = adapter.query_activity_input(
        session, query_id="root", tail="initial-question-marker", output_spec=specs[0], meta={"synthetic": True}
    )
    selected_spec = specs[1] if switch_schema else specs[0]

    @activity.defn(name="synthetic.fresh_review")
    async def review(payload: adapter.AfterQueryInput) -> adapter.AfterQueryDecision:
        nonlocal explicit_ref
        hooks.append(payload)
        if payload.round:
            return adapter.AfterQueryDecision("done")
        if history == "tail":
            return adapter.AfterQueryDecision("follow_up", follow_up_tail="repair-question-marker")
        replacement = replace(
            payload.query,
            query_id="ignored-id",
            round=90,
            tail="repair-question-marker",
            output_name=selected_spec.name,
            output_spec=selected_spec,
        )
        if history == "fresh":
            replacement = replace(
                replacement,
                history_ref="private-must-not-be-read",
                history_json="private-must-not-be-decoded",
            )
        elif history == "explicit":
            assert payload.output.messages_ref is not None
            explicit_ref = await store.put("synthetic-explicit-history", await store.get(payload.output.messages_ref))
            replacement = replace(replacement, history_ref=explicit_ref)
        if history == "legacy":
            return adapter.AfterQueryDecision("follow_up", follow_up_query=replacement)
        mode: Literal["continue", "fresh"] = "fresh" if history == "fresh" else "continue"
        return adapter.AfterQueryDecision("follow_up", follow_up_query=adapter.FollowUpQuery(replacement, mode))

    config = adapter.MapQueriesInput(queries=[query], after_query_activity="synthetic.fresh_review")
    async with running_map(config, review, plugin=plugin) as (env, handle, _queue):
        converter = env.client.data_converter
        wrappers = [adapter.FollowUpQuery(query), adapter.FollowUpQuery(query, "fresh")]
        values = [
            *wrappers,
            *(adapter.AfterQueryDecision("follow_up", follow_up_query=wrapper) for wrapper in wrappers),
            adapter.AfterQueryDecision("follow_up", follow_up_query=query),
        ]
        hints = [adapter.FollowUpQuery, adapter.FollowUpQuery, *([adapter.AfterQueryDecision] * 3)]
        assert await converter.decode(await converter.encode(values), hints) == values
        assert list(asdict(values[-1])) == ["action", "follow_up_tail", "follow_up_query", "result_meta"]
        result = (await handle.result())[0]
        assert result.failure_kind is None
        assert result.query_id == "root#r1"
        assert result.prefix_hash == query.prefix_hash
        assert result.output_type == selected_spec.name
        assert result.output_ref is not None
        assert json.loads(await store.get(result.output_ref)) == {"value": 7 if switch_schema else "OK"}
        assert [payload.query.query_id for payload in hooks] == ["root", "root#r1"]
        assert [payload.round for payload in hooks] == [0, 1]
        assert all(payload.query.prefix_hash == query.prefix_hash for payload in hooks)
        assert [payload.query.output_spec for payload in hooks] == [specs[0], selected_spec]
        assert all(payload.query.history_json is None for payload in hooks)
        assert all(payload.query.require_references for payload in hooks)
        assert all(payload.meta == {"synthetic": True} for payload in hooks)
        if history == "fresh":
            assert hooks[1].query.history_ref is None
            assert "initial-question-marker" not in user_text(requests[1])
        else:
            assert hooks[1].query.history_ref == (
                explicit_ref if history == "explicit" else hooks[0].output.messages_ref
            )
            assert "initial-question-marker" in user_text(requests[1])
        assert "repair-question-marker" in user_text(requests[1])
        assert all(messages[0].parts[0].content == session.rendered_documents for messages in requests)
        assert activity_ids == [adapter.activity_id(query.prefix_hash, query_id) for query_id in ("root", "root#r1")]


@pytest.mark.parametrize("mode", ["raise", "return"])
@pytest.mark.parametrize("invalid_history", ["private-discard", None, 7, []])
async def test_invalid_follow_up_history_is_a_safe_terminal_failure(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["raise", "return"], invalid_history: Any
) -> None:
    query = adapter.QueryActivityInput("prefix", "root", "question", "HookAnswer")
    rounds = []

    async def execute(name, payload, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(payload, adapter.AfterQueryInput):
            assert kwargs["result_type"] is adapter.AfterQueryDecision
            rounds.append(payload.round)
            return adapter.AfterQueryDecision(
                "follow_up",
                follow_up_query=adapter.FollowUpQuery(
                    payload.query, cast(Literal["continue", "fresh"], invalid_history)
                ),
            )
        return adapter.QueryActivityOutput("root", "prefix", None, None, 0, 1, messages_ref="messages")

    monkeypatch.setattr(adapter.workflow, "execute_activity", execute)
    monkeypatch.setattr(adapter.workflow, "info", lambda: SimpleNamespace(parent=None, workflow_id="map"))
    config = adapter.MapQueriesInput(
        queries=[query], after_query_activity="synthetic.invalid_fresh_review", on_query_failure=mode
    )
    result = (await adapter.MapQueriesWorkflow().run(config))[0]
    assert rounds == [0]
    assert result.query_id == "root"
    assert result.failure_kind == "permanent"
    assert result.failure_type == "after_query_failed"
    assert result.messages_ref is None and result.messages_json is None
    assert result.output_ref is None and result.output_json is None
    assert "private" not in json.dumps(asdict(result))
