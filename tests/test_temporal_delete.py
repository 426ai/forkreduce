"""Missing-reference cleanup for real stores and both runnable README recipes."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import sqlite3
from pathlib import Path

import pytest

from forkreduce.contrib._temporal_store import store_get, store_put
from forkreduce.contrib.temporal import InMemoryMessageStore, StoreContext, delete_ref


@pytest.fixture(params=["memory", "README.md", "README.zh-CN.md"])
def store(request: pytest.FixtureRequest, tmp_path: Path):  # type: ignore[no-untyped-def]
    if request.param == "memory":
        return InMemoryMessageStore()

    readme = Path(__file__).resolve().parents[1] / request.param
    blocks = readme.read_text(encoding="utf-8").split("```python\n")[1:]
    sources = [block.split("```", 1)[0] for block in blocks]
    source = next(source for source in sources if "class DurableStore:" in source)
    # Extract the actual documented class without running application setup or
    # the recipe's top-level awaits. Compile its AST, not a duplicate test store.
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DurableStore")
    module = ast.Module(body=[definition], type_ignores=[])
    namespace = {"asyncio": asyncio, "hashlib": hashlib, "sqlite3": sqlite3, "StoreContext": StoreContext}
    exec(compile(module, str(readme), "exec"), namespace)
    return namespace["DurableStore"](str(tmp_path / "blobs.sqlite"))


async def test_delete_missing_and_repeated_refs_preserves_other_blobs(store) -> None:  # type: ignore[no-untyped-def]
    context = StoreContext(kind="messages", prefix_hash="prefix", namespace="tenant/a")
    removed = await store_put(store, "messages:removed", b"remove", context=context)
    retained = await store_put(store, "messages:retained", b"keep", context=context)
    other_context = StoreContext(kind="output", prefix_hash="other-prefix", namespace="tenant/b")
    other = await store_put(store, "output:retained", b"other", context=other_context)

    await delete_ref(store, "tenant%2Fa/messages:never-created", context=context)
    assert await store_get(store, removed, context=context) == b"remove"
    await delete_ref(store, removed, context=context)
    await delete_ref(store, removed, context=context)
    await delete_ref(store, "tenant%2Fa/messages:never-created", context=context)

    with pytest.raises(KeyError):
        await store_get(store, removed, context=context)
    assert await store_get(store, retained, context=context) == b"keep"
    assert await store_get(store, other, context=other_context) == b"other"


async def test_delete_unsupported_store_still_raises() -> None:
    class ReadWriteOnlyStore:
        async def put(self, key: str, payload: bytes) -> str:
            return key

        async def get(self, ref: str) -> bytes:
            raise KeyError(ref)

    with pytest.raises(NotImplementedError, match="does not implement delete"):
        await delete_ref(ReadWriteOnlyStore(), "missing", context=StoreContext(kind="prefix", prefix_hash="prefix"))


async def test_put_preserves_prior_reference_when_logical_key_is_reused(store) -> None:  # type: ignore[no-untyped-def]
    context = StoreContext(kind="messages", prefix_hash="prefix", namespace="tenant")
    first = await store_put(store, "same-query", b"first transcript", context=context)
    second = await store_put(store, "same-query", b"later transcript", context=context)
    assert first != second
    assert await store_get(store, first, context=context) == b"first transcript"
    assert await store_get(store, second, context=context) == b"later transcript"
    await delete_ref(store, second, context=context)
    assert await store_get(store, first, context=context) == b"first transcript"
