"""Worker reconstruction from frozen prefixes and application metadata."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
from temporalio.exceptions import ApplicationError

from forkreduce import build
from forkreduce.contrib import temporal as adapter
from forkreduce.session import Session


class SnapshotAnswer(BaseModel):
    text: str


@pytest.fixture(autouse=True)
def isolated_registrations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "_SESSIONS", {})
    monkeypatch.setattr(adapter, "_FINGERPRINTS", {})
    monkeypatch.setattr(adapter, "_FACTORIES", {})
    monkeypatch.setattr(adapter, "_DEFAULT_FACTORY", None)
    monkeypatch.setattr(adapter, "_STORE", None)
    monkeypatch.setattr(adapter, "_PREFIX_STORE", None)


def make_session(documents: str = "evidence") -> Session:
    return build(
        TestModel(custom_output_args={"text": "restored"}),
        documents=documents,
        instructions="Use the evidence.",
        outputs=[SnapshotAnswer],
        warmup=False,
    )


def payload_for(session: Session, prefix_ref: str | None = None) -> adapter.QueryActivityInput:
    return adapter.query_activity_input(
        session, query_id="q", tail="question", output_name="SnapshotAnswer", prefix_ref=prefix_ref
    )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_restart_with_only_default_factory(asynchronous: bool) -> None:
    store = adapter.InMemoryMessageStore()
    originals = [make_session("first evidence"), make_session("second evidence")]
    queries = [payload_for(session, await adapter.freeze_prefix(session, store)) for session in originals]
    seen: list[str] = []

    def restore(snapshot: adapter.FrozenPrefix) -> Session:
        seen.append(snapshot.prefix_hash)
        return build(
            TestModel(custom_output_args={"text": "restored"}),
            documents=snapshot.documents,
            instructions=snapshot.instructions,
            outputs=[SnapshotAnswer],
            output_mode=snapshot.output_mode,
            on_unforced_tools=snapshot.on_unforced_tools,
            warmup=False,
        )

    async def async_restore(snapshot: adapter.FrozenPrefix) -> Session:
        return restore(snapshot)

    # Simulate a fresh worker: no live sessions or per-prefix configuration.
    del originals
    adapter.set_prefix_store(store)
    adapter.set_default_session_factory(async_restore if asynchronous else restore)
    for query in queries:
        result = await adapter.execute_query_activity(query)
        assert result.output_json is not None
        assert json.loads(result.output_json) == {"text": "restored"}
    assert seen == [query.prefix_hash for query in queries]
    assert not adapter._SESSIONS and not adapter._FACTORIES


async def test_registered_session_then_exact_factory_then_default_precedence() -> None:
    session = make_session()
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store)
    adapter.set_prefix_store(store)
    calls: list[str] = []

    def exact(_snapshot: adapter.FrozenPrefix) -> Session:
        calls.append("exact")
        return session

    def default(_snapshot: adapter.FrozenPrefix) -> Session:
        calls.append("default")
        return session

    adapter.set_default_session_factory(default)
    adapter.register_session_factory(session.prefix_hash, exact)
    adapter.register_session(session)
    assert await adapter._resolve_session(payload_for(session)) is session
    assert not calls
    adapter.unregister_session(session.prefix_hash)
    adapter.register_session_factory(session.prefix_hash, exact)
    assert await adapter._resolve_session(payload_for(session, ref)) is session
    adapter.unregister_session(session.prefix_hash)
    assert await adapter._resolve_session(payload_for(session, ref)) is session
    assert calls == ["exact", "default"]


@pytest.mark.parametrize("with_reference", [False, True])
async def test_missing_factory_is_retryable_and_default_can_be_cleared(with_reference: bool) -> None:
    session = make_session()
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store) if with_reference else None
    adapter.set_prefix_store(store)
    adapter.set_default_session_factory(lambda _snapshot: session)
    adapter.set_default_session_factory(None)
    with pytest.raises(ApplicationError) as caught:
        await adapter.execute_query_activity(payload_for(session, ref))
    assert caught.value.type == "session_factory_missing"
    assert caught.value.non_retryable is False


@pytest.mark.parametrize("exact", [False, True])
async def test_factory_prefix_mismatch_remains_permanent(exact: bool) -> None:
    session = make_session()
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store)
    adapter.set_prefix_store(store)
    adapter.set_default_session_factory(lambda _snapshot: make_session("wrong evidence"))
    if exact:
        # An invalid exact factory must not fall back to a valid default.
        adapter.register_session_factory(session.prefix_hash, lambda _snapshot: make_session("wrong evidence"))
        adapter.set_default_session_factory(lambda _snapshot: session)
    with pytest.raises(ApplicationError, match="prefix hash") as caught:
        await adapter.execute_query_activity(payload_for(session, ref))
    assert caught.value.non_retryable is True
    assert caught.value.type == "permanent"


async def test_default_factory_requires_snapshot_reference() -> None:
    session = make_session()
    adapter.set_default_session_factory(lambda _snapshot: session)
    with pytest.raises(ApplicationError, match="requires prefix_ref") as caught:
        await adapter.execute_query_activity(payload_for(session))
    assert caught.value.non_retryable is True
    assert caught.value.type == "permanent"
