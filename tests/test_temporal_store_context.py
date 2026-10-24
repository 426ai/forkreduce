"""Storage routing, backward compatibility and explicit retention boundaries."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
from temporalio.converter import DataConverter
from temporalio.testing import ActivityEnvironment

from forkreduce import build
from forkreduce.contrib import temporal as adapter


class StoredAnswer(BaseModel):
    text: str


class ContextStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.calls: list[tuple[str, str, adapter.StoreContext]] = []

    async def put(self, key: str, payload: bytes, *, context: adapter.StoreContext) -> str:
        self.calls.append(("put", key, context))
        self.data[key] = payload
        return key

    async def get(self, ref: str, *, context: adapter.StoreContext) -> bytes:
        self.calls.append(("get", ref, context))
        return self.data[ref]

    async def delete(self, ref: str, *, context: adapter.StoreContext) -> None:
        self.calls.append(("delete", ref, context))
        del self.data[ref]


def session():  # type: ignore[no-untyped-def]
    return build(
        TestModel(custom_output_args={"text": "stored"}),
        documents="Synthetic storage evidence.",
        instructions="Use the evidence.",
        outputs=[StoredAnswer],
        warmup=False,
    )


@pytest.fixture(autouse=True)
def isolated(monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(adapter, "_SESSIONS", {})
    monkeypatch.setattr(adapter, "_FINGERPRINTS", {})
    monkeypatch.setattr(adapter, "_FACTORIES", {})
    monkeypatch.setattr(adapter, "_DEFAULT_FACTORY", None)
    monkeypatch.setattr(adapter, "_STORE", None)
    monkeypatch.setattr(adapter, "_PREFIX_STORE", None)


async def test_namespaces_separate_identical_prefixes_and_preserve_legacy_keys() -> None:
    original = session()
    store = ContextStore()
    first = await adapter.freeze_prefix(original, store, app_meta={"ref_namespace": "tenant/a"})
    second = await adapter.freeze_prefix(original, store, app_meta={"ref_namespace": "tenant/b"})
    legacy = await adapter.freeze_prefix(original, store)
    assert first == f"tenant%2Fa/prefix:{original.prefix_hash}"
    assert second == f"tenant%2Fb/prefix:{original.prefix_hash}"
    assert legacy == f"prefix:{original.prefix_hash}"
    assert json.loads(store.data[first])["app_meta"] == {"ref_namespace": "tenant/a"}
    assert json.loads(store.data[second])["app_meta"] == {"ref_namespace": "tenant/b"}
    assert store.calls[0][2] == adapter.StoreContext(
        kind="prefix", prefix_hash=original.prefix_hash, namespace="tenant/a"
    )


@pytest.mark.parametrize("override", [None, "query-space"])
async def test_context_across_prefix_history_messages_output_and_delete(override: str | None) -> None:
    original = session()
    store = ContextStore()
    ref = await adapter.freeze_prefix(original, store, app_meta={"ref_namespace": "snapshot-space"})
    adapter.set_prefix_store(store)
    adapter.set_message_store(store)
    adapter.set_default_session_factory(lambda _snapshot: original)
    query = adapter.query_activity_input(
        original,
        query_id="q1",
        tail="First question",
        output_name="StoredAnswer",
        prefix_ref=ref,
        meta={} if override is None else {"ref_namespace": override},
    )
    query.require_references = True
    query.parent_workflow_id = "parent"
    env = ActivityEnvironment()
    env.info = replace(env.info, workflow_id="child", workflow_run_id="run", activity_id="activity", attempt=2)
    first = await env.run(adapter.execute_query_activity, query)
    assert first.messages_ref is not None and first.output_ref is not None
    namespace = override or "snapshot-space"
    assert first.messages_ref == f"{namespace}/{original.prefix_hash}:q1"
    assert first.output_ref == f"{namespace}/output:{original.prefix_hash}:q1"
    for operation, _ref, context in store.calls[1:]:
        assert context.workflow_id == "child"
        assert context.run_id == "run"
        assert context.parent_workflow_id == "parent"
        assert context.activity_id == "activity"
        assert context.query_id == "q1"
        assert context.round == 0 and context.attempt == 2
        assert context.prefix_hash == original.prefix_hash
        assert context.namespace == (override if context.kind == "prefix" else namespace)
        assert operation == ("get" if context.kind == "prefix" else "put")
    await env.run(
        adapter.execute_query_activity,
        replace(query, query_id="q2", tail="Continue", history_ref=first.messages_ref, round=1),
    )
    history_context = next(context for _, _, context in store.calls if context.kind == "history")
    assert history_context.query_id == "q2" and history_context.round == 1
    assert history_context.namespace == namespace
    # Nothing was removed by either activity. Application alone owns retention.
    assert set(store.data) >= {ref, first.messages_ref, first.output_ref}
    context = adapter.StoreContext(kind="output", prefix_hash=original.prefix_hash, namespace=namespace)
    await adapter.delete_ref(store, first.output_ref, context=context)
    assert first.output_ref not in store.data
    assert store.calls[-1] == ("delete", first.output_ref, context)
    encoded = await DataConverter.default.encode([context])
    assert await DataConverter.default.decode(encoded, [adapter.StoreContext]) == [context]


async def test_legacy_store_supports_namespaced_freeze_activity_and_explicit_delete() -> None:
    class LegacyStore:
        def __init__(self):  # type: ignore[no-untyped-def]
            self.data = {}

        async def put(self, key, payload):  # type: ignore[no-untyped-def]
            self.data[key] = payload
            return key

        async def get(self, ref):  # type: ignore[no-untyped-def]
            return self.data[ref]

        async def delete(self, ref):  # type: ignore[no-untyped-def]
            del self.data[ref]

    original = session()
    store = LegacyStore()
    ref = await adapter.freeze_prefix(original, store, app_meta={"ref_namespace": "legacy"})
    adapter.set_prefix_store(store)
    adapter.set_message_store(store)
    adapter.set_default_session_factory(lambda _snapshot: original)
    result = await adapter.execute_query_activity(
        replace(
            adapter.query_activity_input(
                original, query_id="q", tail="Question", output_name="StoredAnswer", prefix_ref=ref
            ),
            require_references=True,
        )
    )
    assert result.output_ref == f"legacy/output:{original.prefix_hash}:q"
    assert json.loads(await store.get(result.output_ref)) == {"text": "stored"}
    await adapter.delete_ref(store, ref, context=adapter.StoreContext(kind="prefix", prefix_hash=original.prefix_hash))
    assert ref not in store.data
    with pytest.raises(NotImplementedError):
        await adapter.delete_ref(object(), "ref", context=adapter.StoreContext(kind="prefix", prefix_hash="hash"))


async def test_store_type_error_does_not_repeat_write() -> None:
    class BrokenStore(ContextStore):
        writes = 0

        async def put(self, key: str, payload: bytes, *, context: adapter.StoreContext) -> str:
            self.writes += 1
            raise TypeError("Storage implementation failed after writing")

    store = BrokenStore()
    with pytest.raises(TypeError, match="after writing"):
        await adapter.freeze_prefix(session(), store)
    assert store.writes == 1
