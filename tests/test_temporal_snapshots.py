"""Worker reconstruction from frozen prefixes and application metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from types import MappingProxyType

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


@pytest.mark.parametrize("exact", [False, True])
async def test_factory_receives_independent_application_metadata(exact: bool) -> None:
    session = make_session()
    prefix_hash = session.prefix_hash
    fingerprint = adapter.execution_fingerprint(session)
    metadata: dict[str, adapter.JSONValue] = {
        "deployment": "offline",
        "options": {"labels": ["original"], "enabled": True, "limit": 3, "ratio": 0.5, "extra": None},
    }
    original = json.loads(json.dumps(metadata))
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store, app_meta=MappingProxyType(metadata))
    options = metadata["options"]
    assert isinstance(options, dict)
    options["labels"] = ["caller changed"]
    metadata["deployment"] = "caller changed"
    seen: list[dict[str, adapter.JSONValue]] = []

    def restore(snapshot: adapter.FrozenPrefix) -> Session:
        assert snapshot.app_meta == original
        snapshot.validate()
        seen.append(snapshot.app_meta)
        snapshot.app_meta["deployment"] = "factory changed"
        nested = snapshot.app_meta["options"]
        assert isinstance(nested, dict)
        nested["labels"] = ["factory changed"]
        return session

    adapter.set_prefix_store(store)
    if exact:
        adapter.register_session_factory(session.prefix_hash, restore)
    else:
        adapter.set_default_session_factory(restore)
    for _ in range(2):
        assert await adapter._resolve_session(payload_for(session, ref)) is session
    assert seen[0] is not seen[1]
    assert metadata["deployment"] == "caller changed"
    assert options["labels"] == ["caller changed"]
    assert session.prefix_hash == prefix_hash
    assert adapter.execution_fingerprint(session) == fingerprint
    frozen = adapter.FrozenPrefix(**json.loads(await store.get(ref)))
    assert frozen.prefix_hash == prefix_hash and frozen.app_meta == original


async def test_legacy_and_default_metadata_dicts_are_independent() -> None:
    session = make_session()
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store)
    record = json.loads(await store.get(ref))
    assert record.pop("app_meta") == {}
    first = adapter.FrozenPrefix(**record)
    second = adapter.FrozenPrefix(**record)
    first.app_meta["only_first"] = True
    assert second.app_meta == {}
    first.validate()
    second.validate()
    assert first.prefix_hash == second.prefix_hash == session.prefix_hash
    legacy_ref = await store.put("legacy", json.dumps(record).encode())
    seen: list[dict[str, adapter.JSONValue]] = []

    def restore(snapshot: adapter.FrozenPrefix) -> Session:
        seen.append(snapshot.app_meta)
        return session

    adapter.set_prefix_store(store)
    adapter.set_default_session_factory(restore)
    assert await adapter._resolve_session(payload_for(session, legacy_ref)) is session
    assert seen == [{}]
    explicit_ref = await adapter.freeze_prefix(session, store, app_meta=None)
    assert json.loads(await store.get(explicit_ref))["app_meta"] == {}


@pytest.mark.parametrize("character", ["x", "é", "界", "🦊"])
async def test_metadata_limit_counts_compact_utf8_bytes(character: str) -> None:
    session = make_session()
    store = adapter.InMemoryMessageStore()
    limit = 16 * 1024
    overhead = len(json.dumps({"text": ""}, separators=(",", ":")).encode("utf-8"))
    count, remainder = divmod(limit - overhead, len(character.encode("utf-8")))
    metadata: dict[str, adapter.JSONValue] = {"text": character * count + "x" * remainder}
    compact = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(compact) == limit
    ref = await adapter.freeze_prefix(session, store, app_meta=metadata)
    frozen = adapter.FrozenPrefix(**json.loads(await store.get(ref)))
    frozen.validate()
    assert frozen.app_meta == metadata
    assert character.encode("utf-8") in await store.get(ref)
    assert isinstance(metadata["text"], str)
    metadata["text"] += "x"
    with pytest.raises(ValueError, match="16 KiB"):
        await adapter.freeze_prefix(session, store, app_meta=metadata)
    # Validation must also reject oversized metadata attached to a valid prefix.
    record = asdict(frozen)
    record["app_meta"] = metadata
    with pytest.raises(ValueError, match="16 KiB"):
        adapter.FrozenPrefix(**record).validate()


@pytest.mark.parametrize(
    "metadata",
    [
        {"invalid": object()},
        {"invalid": (1, 2)},
        {1: "non-string key"},
        {"invalid": float("nan")},
        {"invalid": float("inf")},
        ["not an object"],
    ],
)
async def test_freeze_rejects_non_json_metadata(metadata: Mapping[str, adapter.JSONValue]) -> None:
    with pytest.raises((ValueError, TypeError)):
        await adapter.freeze_prefix(make_session(), adapter.InMemoryMessageStore(), app_meta=metadata)


@pytest.mark.parametrize("metadata", [None, ["not an object"], {"text": "x" * (16 * 1024)}, {"bad": float("nan")}])
async def test_worker_validates_metadata_before_factory(metadata: adapter.JSONValue) -> None:
    session = make_session()
    store = adapter.InMemoryMessageStore()
    ref = await adapter.freeze_prefix(session, store)
    record = json.loads(await store.get(ref))
    record["app_meta"] = metadata
    corrupt_ref = await store.put("corrupt", json.dumps(record).encode())
    calls: list[str] = []

    def restore(snapshot: adapter.FrozenPrefix) -> Session:
        calls.append(snapshot.prefix_hash)
        return session

    adapter.set_prefix_store(store)
    adapter.set_default_session_factory(restore)
    with pytest.raises(ApplicationError) as caught:
        await adapter.execute_query_activity(payload_for(session, corrupt_ref))
    assert caught.value.non_retryable is True
    assert caught.value.type == "permanent"
    assert not calls
