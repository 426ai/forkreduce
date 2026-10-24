# Changelog

## 0.2.0 — 2026-10-25

### Added

- SDK-independent, JSON-serializable `ActivityOptions`, available on `build`, individual Temporal query payloads, and `MapQueriesInput`: start-to-close, heartbeat, schedule-to-close, total attempts, and exponential-backoff settings. Defaults remain five minutes and `Policy.retries + 1` attempts. Configured heartbeats cover the full asynchronous query activity, with cancellation cleanup.
- Optional bounded child batching via `MapQueriesInput(batch_size=..., child_concurrency=...)` and `QueryBatchWorkflow`. Ordered results and deterministic child IDs include parent workflow ID, prefix hash, and batch index; IDs survive workflow retries that change run ID. Single-workflow/list-input mode stays the default.
- `PrefixStore`, `FrozenPrefix`, `freeze_prefix`, `set_prefix_store`, and worker-local `register_session_factory` support document reconstruction from references. Batched mode requires external transcript/result storage and returns `messages_ref` / `output_ref`, keeping transcript and structured-output bodies out of parent history. The in-memory store remains test-only, not a durable backend.
- Real time-skipping workflow tests for the default converter and `PydanticAIPlugin` / `PydanticAIPayloadConverter`, including offline `TestModel`/`FunctionModel`, shared-worker `TemporalDurability`, replay, bounded children, heartbeat/retry behavior, and large-document payload bounds.
- Opt-in `Citation` schema and `CITATION_INSTRUCTIONS`: a contiguous quote must fit inside its single referenced block; split cross-block evidence into separate citations. Citation validators remain strict and unchanged.
- Capability-aware `output_mode="auto"` (the new build default), requested `Session.output_mode`, actual `Session.resolved_output_mode`, and `QueryResult.resolved_output_mode` (including failures/reduce combinations; legacy default `None`). Auto resolves from public model profile fields and shallow-merged model/build settings, accounting for thinking restrictions; it never selects native output or retries in a different mode.
- `on_unforced_tools="warn"|"error"|"ignore"` for explicit tools without effective forcing support (`UnforcedToolsWarning` / `UnforcedToolsError`), including capability-changing query settings. Auto query overrides that would flip the resolved mode raise `OutputModeConflictError` before budget/provider use; build a new session instead.
- When thinking can affect forcing capability, auto rejects unknown/malformed non-null thinking controls with `ForkReduceError` code `unknown_thinking_control`; explicit tools uses the unforced-tools policy, while explicit prompted remains honored. Unconditionally supported/unsupported profiles resolve directly without rejecting irrelevant thinking controls. Recognized OpenAI, Anthropic, Google, and common `extra_body` controls take precedence over unified thinking, with conflicting controls conservatively active.
- `FrozenPrefix` retains requested mode, optional resolved mode for legacy snapshots, and warning policy. Factories restore the requested mode/policy and matching thinking-related settings; workers reject resolved-mode/hash mismatches.

### Changed

- Reduce tails explicitly enumerate allowed input/fork IDs and per-question source mappings when `question_id` fields are present, including nested outputs and tree aliases. Document/block IDs are not reduce provenance IDs. Citation-repair tails repeat the single-block constraint.
- English and Chinese documentation include the 2-attempt / 3-minute / 30-second-heartbeat recipe, plugin coexistence, batching/reference storage, and reasoning-token headroom. Primary-source links explain model/vendor forced-tool restrictions (HTTP 400 or silent `auto`, especially with thinking), exact Pydantic AI 2.61 profile semantics/defaults, and MiMo thinking's forced `temperature=1.0` / `top_p=0.95` despite requested zero temperature.

### Fixed

- Keep a single periodic heartbeat task alive across prefix hydration, continuation-history reads, async session-factory construction, model calls, and transcript/structured-output uploads. Previously heartbeats stopped outside the model await, so a healthy slow store could time out and repeat a successful model call on retry. Real-worker regressions cover each reference I/O boundary and the async factory with a 200 ms timeout, a 650 ms wait, and exactly one attempt under both converters. Blocking the event loop remains outside heartbeat protection.

### Compatibility and cache identity

- No prefix-hash algorithm change: hashes use actual tools/prompted mode, never an auto marker. Auto resolving to tools preserves the previous default tools hash; unsupported forcing now defaults to prompted and changes that old default hash. Explicit tools/prompted hashes remain stable. Thinking settings/profile changes can change auto resolution; per-query changes cannot flip it. Reduce/repair guidance changes only tails. Adopting `Citation` or `CITATION_INSTRUCTIONS` changes schema/instruction bytes and therefore `prefix_hash` (including prompted schema fingerprints); updated examples consequently miss old provider prefix caches. `ActivityOptions`, batching, and reference storage do not change prefix identity.
- Activity/workflow names and legacy list inputs remain supported. Batch results intentionally use references rather than inline bodies. Child IDs identify orchestration, not exactly-once provider calls; retries can repeat external effects. Retain referenced data through retries/continuations. Drain or version incompatible in-flight workflows and replay representative histories before changing deployment code/converters.
- Verified dependency targets: Pydantic AI 2.61.0, OpenAI 3.37.0, Temporal SDK 1.33.0 and 1.34.0. Core requirement remains `pydantic-ai>=2.57,<3`; the adapter does not require raising its existing `temporalio>=1.9` floor. Official Pydantic AI plugin users must also satisfy that integration's own `[temporal]` requirements.
- Version metadata is prepared as 0.2.0. This preparation does not publish the package, create a tag/release, merge the PR, or change deployments.

### Upgrade notes

- **Cache identity:** the hash algorithm is unchanged. For unchanged schemas/instructions, the new default `output_mode="auto"` preserves the 0.1.1 tools hash when the effective profile supports forced output tools. Users whose profile forbids forcing, or whose active thinking prevents forcing, now default to prompted and get a different prefix/hash and a cold provider cache. Explicit tools/prompted retain their previous hashes. Adopting `Citation`, `CITATION_INSTRUCTIONS`, changed output descriptions, or the updated example schemas/instructions also changes hashes. Auto mode is frozen per session; a query thinking override that would flip it raises `OutputModeConflictError` before provider use—build a separate session.
- **In-flight Temporal executions:** before deploying 0.2.0, stop admitting new work to the old task queue and let its running workflows/children/activities finish on the old worker code and converter. Keep referenced blobs available through retries and continuations. Then switch clients/workers together. If executions cannot be drained, isolate the new code on a new task queue or compatible worker version and replay representative old histories before routing existing work to it; converter roundtrips alone do not prove arbitrary old workflow-code replay compatibility.
- **Worker sandbox:** register both `MapQueriesWorkflow` and `QueryBatchWorkflow` for batching. A default-converter worker must pass `SandboxedWorkflowRunner(restrictions=SandboxRestrictions.default.with_passthrough_modules("forkreduce", "pydantic_ai"))` as its `workflow_runner`; both classes are in `temporalio.worker.workflow_sandbox`. Apply the same runner configuration when replaying histories. With `PydanticAIPlugin`, install Pydantic AI's `[temporal]` dependencies and inherit the client plugin on the worker rather than installing it twice.
- **Heartbeat limits:** set a finite start-to-close timeout in addition to heartbeat_timeout. The full-activity heartbeat protects cooperative async waits, including shared-store I/O, not CPU work blocking the event loop; offload that work to a thread or a separate activity. Heartbeats signal liveness, not business progress or exactly-once external effects.

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
