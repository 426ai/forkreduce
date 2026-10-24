"""Storage contracts and signature-based compatibility for the Temporal adapter."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast
from urllib.parse import quote

from temporalio import activity

StoreKind = Literal["prefix", "messages", "output", "history"]


@dataclass(frozen=True)
class StoreContext:
    """JSON-native routing metadata; missing execution fields remain ``None``.

    ``namespace`` is application routing metadata, not the Temporal namespace.
    A prefix read can only use metadata already present on the query; snapshot
    metadata becomes available after hydration. Repeat ``ref_namespace`` in the
    query metadata when the store needs it to authorize the initial read.
    """

    kind: StoreKind
    prefix_hash: str
    workflow_id: str | None = None
    run_id: str | None = None
    parent_workflow_id: str | None = None
    activity_id: str | None = None
    query_id: str | None = None
    round: int | None = None
    attempt: int | None = None
    namespace: str | None = None


class MessageStore(Protocol):
    """Legacy store contract, still accepted without a context argument."""

    async def put(self, key: str, payload: bytes) -> str: ...

    async def get(self, ref: str) -> bytes: ...


class PrefixStore(MessageStore, Protocol):
    """Legacy prefix storage contract."""


class ContextMessageStore(Protocol):
    """Context-aware store; deletion is optional and detected separately."""

    async def put(self, key: str, payload: bytes, *, context: StoreContext) -> str: ...

    async def get(self, ref: str, *, context: StoreContext) -> bytes: ...


Store = MessageStore | ContextMessageStore


def ref_namespace(meta: Mapping[str, Any]) -> str | None:
    """Read an optional namespace without coercing other JSON values to strings."""
    value = meta.get("ref_namespace")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("ref_namespace must be a string")
    return value or None


def store_context(
    kind: StoreKind,
    prefix_hash: str,
    *,
    namespace: str | None = None,
    query_id: str | None = None,
    round: int | None = None,
    parent_workflow_id: str | None = None,
) -> StoreContext:
    info = activity.info() if activity.in_activity() else None
    return StoreContext(
        kind=kind,
        prefix_hash=prefix_hash,
        workflow_id=info.workflow_id if info is not None else None,
        run_id=info.workflow_run_id if info is not None else None,
        parent_workflow_id=parent_workflow_id,
        activity_id=info.activity_id if info is not None else None,
        query_id=query_id,
        round=round,
        attempt=info.attempt if info is not None else None,
        namespace=namespace,
    )


def _accepts_context(method: Callable[..., Any]) -> bool:
    # Detect before invoking: retrying after TypeError could duplicate a write.
    parameters = inspect.signature(method).parameters
    context = parameters.get("context")
    return (
        context is not None
        and context.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ) or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


async def store_put(store: Store, key: str, payload: bytes, *, context: StoreContext) -> str:
    if context.namespace:
        key = f"{quote(context.namespace, safe='')}/{key}"
    method = cast(Callable[..., Awaitable[str]], store.put)
    if _accepts_context(method):
        return await method(key, payload, context=context)
    return await method(key, payload)


async def store_get(store: Store, ref: str, *, context: StoreContext) -> bytes:
    method = cast(Callable[..., Awaitable[bytes]], store.get)
    if _accepts_context(method):
        return await method(ref, context=context)
    return await method(ref)


async def delete_ref(store: Store, ref: str, *, context: StoreContext) -> None:
    """Delete only on application request; unsupported stores raise NotImplementedError.

    Supply the original reference's kind, prefix hash and namespace in ``context``.
    No framework path calls this helper automatically.
    """
    method = getattr(store, "delete", None)
    if not callable(method):
        raise NotImplementedError("This store does not implement delete")
    if _accepts_context(method):
        await method(ref, context=context)
    else:
        await method(ref)
