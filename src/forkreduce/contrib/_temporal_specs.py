"""Content-addressed OutputSpec preparation; schemas never enter workflow history."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass

from forkreduce.contrib._temporal_store import Store, store_context, store_put
from forkreduce.output_specs import OutputSpec


@dataclass(frozen=True)
class OutputSpecRef:
    """Immutable schema identity and opaque store reference, without schema bytes."""

    name: str
    ref: str
    sha256: str
    namespace: str | None = None


def encode_spec(spec: OutputSpec, namespace: str | None) -> bytes:
    return json.dumps(
        {"namespace": namespace, "spec": asdict(spec)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class OutputSpecWriter:
    """Prepare specs in an existing store, deduplicating within this writer's lifetime.

    Keep one writer for a preparation job. Equal specs in the same namespace use
    one put, even concurrently, including stores returning opaque/random refs.
    Content-addressed backends can also deduplicate across writers/processes.
    The writer/cache is client/activity-owned and is never passed to workflows.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self._refs: dict[tuple[str | None, str], OutputSpecRef] = {}
        self._lock = asyncio.Lock()

    async def store(self, spec: OutputSpec, *, ref_namespace: str | None = None) -> OutputSpecRef:
        """Store exact JSON schema bytes outside workflows; namespace is routing, not authorization."""
        if ref_namespace is not None and not isinstance(ref_namespace, str):
            raise ValueError("ref_namespace must be a string")
        namespace = ref_namespace or None
        encoded = encode_spec(spec, namespace)
        digest = hashlib.sha256(encoded).hexdigest()
        key = (namespace, digest)
        async with self._lock:
            existing = self._refs.get(key)
            if existing is not None:
                return existing
            ref = await store_put(
                self._store,
                f"output-spec:{digest}",
                encoded,
                context=store_context("output_spec", "", namespace=namespace),
            )
            result = OutputSpecRef(spec.name, ref, digest, namespace)
            self._refs[key] = result
            return result
