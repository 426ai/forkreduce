# Changelog

## 0.2.0 — Unreleased

### Added

- SDK-independent, JSON-serializable `ActivityOptions`, available on `build`, individual Temporal query payloads, and `MapQueriesInput`: start-to-close, heartbeat, schedule-to-close, total attempts, and exponential-backoff settings. Defaults remain five minutes and `Policy.retries + 1` attempts. Configured heartbeats continue while awaiting the provider, with cancellation cleanup.
- Optional bounded child batching via `MapQueriesInput(batch_size=..., child_concurrency=...)` and `QueryBatchWorkflow`. Ordered results and deterministic child IDs include parent workflow ID, prefix hash, and batch index; IDs survive workflow retries that change run ID. Single-workflow/list-input mode stays the default.
- `PrefixStore`, `FrozenPrefix`, `freeze_prefix`, `set_prefix_store`, and worker-local `register_session_factory` support document reconstruction from references. Batched mode requires external transcript/result storage and returns `messages_ref` / `output_ref`, keeping transcript and structured-output bodies out of parent history. The in-memory store remains test-only, not a durable backend.
- Real time-skipping workflow tests for the default converter and `PydanticAIPlugin` / `PydanticAIPayloadConverter`, including offline `TestModel`/`FunctionModel`, shared-worker `TemporalDurability`, replay, bounded children, heartbeat/retry behavior, and large-document payload bounds.
- Opt-in `Citation` schema and `CITATION_INSTRUCTIONS`: a contiguous quote must fit inside its single referenced block; split cross-block evidence into separate citations. Citation validators remain strict and unchanged.

### Changed

- Reduce tails explicitly enumerate allowed input/fork IDs and per-question source mappings when `question_id` fields are present, including nested outputs and tree aliases. Document/block IDs are not reduce provenance IDs. Citation-repair tails repeat the single-block constraint.
- English and Chinese documentation include the 2-attempt / 3-minute / 30-second-heartbeat recipe, plugin coexistence, batching/reference storage, prompted-output fallback for gateways ignoring required tool choice, and reasoning-token headroom.

### Compatibility and cache identity

- No prefix-hash algorithm change. Existing user schemas/instructions keep their hashes: reduce and repair guidance changes only query tails. Adopting `Citation` or adding `CITATION_INSTRUCTIONS` changes schema/instruction bytes and therefore `prefix_hash` (including prompted mode's schema fingerprint). The updated examples consequently have new hashes and miss the old provider prefix cache on upgrade. `ActivityOptions`, batching, and reference storage do not change prefix identity.
- Activity/workflow names and legacy list inputs remain supported. Batch results intentionally use references rather than inline bodies. Child IDs identify orchestration, not exactly-once provider calls; retries can repeat external effects. Retain referenced data through retries/continuations. Drain or version incompatible in-flight workflows and replay representative histories before changing deployment code/converters.
- Verified dependency targets: Pydantic AI 2.61.0, OpenAI 3.37.0, Temporal SDK 1.33.0 and 1.34.0. Core requirement remains `pydantic-ai>=2.57,<3`; the adapter does not require raising its existing `temporalio>=1.9` floor. Official Pydantic AI plugin users must also satisfy that integration's own `[temporal]` requirements.
- This section is unreleased; package metadata remains at the last published version until release preparation. No tag or deployment change.

## 0.1.1

Backward compatible. Calls that do not opt in to rendering `Block.meta` keep the same prefix hash as 0.1.0.

### Added

- `openai_compatible_model()` accepts `timeout`, `profile` (a pydantic-ai `ModelProfile`), `extra_body`, and the common sampling settings (`temperature`, `top_p`, `max_tokens`, `seed`, penalties, `stop_sequences`, `parallel_tool_calls`, `extra_headers`). A `settings` dict passes any other `ModelSettings` key. Explicit arguments override `settings`.
- `build(..., model_settings=...)` sends those pydantic-ai `ModelSettings` on every call. `Query.model_settings` overrides individual keys for one call. The merge is shallow, so a query `extra_body` replaces the session `extra_body` as a whole.
- The Temporal execution fingerprint includes request-body settings and the resolved model profile. `timeout` and `extra_headers` are transport options and are not part of the fingerprint. A per-query override is not part of session identity; `QueryActivityInput.model_settings` carries it to the worker.
- Optional `Block.meta` rendering via `build(..., meta_keys=[...])` or `meta_renderer=...`. Rendered meta is part of the prefix and `prefix_hash`. The default still omits meta.

Settings that change prompt or tool bytes (`meta_keys`, `meta_renderer`, and a `profile` that changes tool or schema rendering) miss the provider prefix cache. Sampling parameters, `timeout`, `extra_headers`, and `extra_body` do not change the frozen prefix. `extra_body` and sampling parameters still change the request body, so a different value is a different execution fingerprint.

## 0.1.0

Initial public release.
