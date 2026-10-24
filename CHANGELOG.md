# Changelog

## Unreleased

- Add a source-tree-only CUAD benchmark with three method adapters, offline FunctionModel/TestModel coverage, provider-response usage accounting, span/citation metrics, contract-cluster confidence intervals, and real-data cost estimation. Paid execution requires separate estimate approval; benchmark dependencies and downloaded data stay outside the core wheel.
- Add an explicitly approved minimal multi-model smoke runner with Chat/Responses selection, configurable cache keys/headers and capability profiles, normalized cached/reasoning usage, persistent per-request spend reservations, and attempt-aware reports. Fail closed on unbounded billable output: xAI's visible-only cap cannot enforce a total-token budget; unsafe models remain isolated without discarding their costs or blocking independently bounded providers.
- Add opt-in bounded, credential-redacted failed-response diagnostics and benchmark-only tools/prompted output selection, retaining strict question-ID, provenance, and citation validation. Add a two-contract MiMo/DeepSeek diagnostic CLI with a separate persistent $0.60 request ledger; ordinary usage now includes safe finish-reason and tool-call summaries. Core forkreduce API semantics are unchanged.
- Configure MiMo CUAD forkreduce to use prompted JSON and 8,192 output tokens, with the same cap for one-shot; give DeepSeek forkreduce the same reasoning headroom. Harden generic benchmark provenance mappings and citation block-boundary guidance without relaxing validators or changing core request/retry semantics.

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
