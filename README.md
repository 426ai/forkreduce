<p align="center">
  <img src="https://github.com/426ai/forkreduce/raw/main/docs/assets/logo.svg" alt="forkreduce logo" width="160">
</p>
<h1 align="center">forkreduce</h1>

forkreduce is a small Python framework, built on [Pydantic AI](https://ai.pydantic.dev/), for the case where many LLM queries share one long context.

![forkreduce overview: reuse one stable prefix of tools, instructions, and documents across parallel queries; check, repair, and filter results, then merge results while retaining provenance. Reports retain gaps and failures, and optional follow-ups start another round. Provider cache hits are not guaranteed.](https://github.com/426ai/forkreduce/raw/main/docs/assets/overview.svg)

## Installation

Requires Python 3.11 or newer.

```bash
pip install forkreduce
```

The Temporal adapter is optional: `pip install "forkreduce[temporal]"`. The core package never imports it.

## Quick start

Ask two questions about the same document (map), then merge the answers into a summary with provenance (reduce). This example calls a real model; first configure an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="your-model-name"
# Optional for a custom endpoint:
# export OPENAI_BASE_URL="https://your-provider.example/v1"
```

```python
import asyncio
from pydantic import BaseModel
from forkreduce import Query, Traced, build, openai_compatible_model


class Answer(BaseModel):
    text: str


class Summary(Traced):
    text: str


async def main() -> None:
    session = build(
        openai_compatible_model(),
        documents="Eligible titles: Science Digest and River Almanac. Deadline: 30 April 2026.",
        instructions="Answer only from the document.",
        outputs=[Answer, Summary],
    )
    result = await (
        session.map([
            Query(id="titles", tail="Which titles are eligible?", output=Answer),
            Query(id="deadline", tail="What is the deadline?", output=Answer),
        ])
        .reduce(output=Summary, instructions="Combine both answers into one concise summary.")
        .run()
    )
    if result.merged is not None:
        print(result.merged.text)
    if result.failures:
        print(result.failures)


asyncio.run(main())
```

Example output (wording depends on the model):

```text
Eligible titles are Science Digest and River Almanac; the deadline is 30 April 2026.
```

Provide the documents and instructions once at `build()`. Map and reduce share that prefix; only their query tails change. Register both output types up front; `Summary` inherits provenance tracking from `Traced`, and the merged result is available as `result.merged`. By default, the first map query finishes as a warmup before the rest run concurrently. Actual cache hits depend on the provider. For an offline example with no API key, see the `TestModel` example below.

## Full workflow example

Add citation checks, bounded repair, a business filter, and a merge. `TestModel` generates test data to demonstrate the API flow, not real document extraction.

The notice below is fictional. Northbridge is not a real city, and the titles and quantities are invented.

```python
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai.models.test import TestModel

from forkreduce import (
    CITATION_INSTRUCTIONS,
    Block,
    Citation,
    CitationCheck,
    CitationRepair,
    Document,
    Query,
    Traced,
    build,
)


class Answer(Citation):
    """One fact from the notice."""

    text: str


class Finding(Traced):
    topic: str
    summary: str
    source_ids: list[str] = Field(default_factory=list)


class FindingList(BaseModel):
    """Merged findings."""

    items: list[Finding]


notice = Document(
    doc_id="nb-lib-2026-per",
    title="Northbridge Public Library 2026 periodical subscription notice",
    blocks=[
        Block(block_id="titles", text="Eligible titles are the Northbridge Science Digest and the River Almanac."),
        Block(block_id="deadline", text="Offers must reach the front desk by 30 April 2026."),
    ],
)


async def main() -> None:
    # TestModel needs no network. Pass any pydantic-ai Model in production.
    session = build(
        TestModel(custom_output_args={"text": "see notice", "quote": "Eligible titles", "block_id": "titles"}),
        documents=notice,
        instructions="Answer only from the notice. " + CITATION_INSTRUCTIONS,
        outputs=[Answer, FindingList],
    )
    result = await (
        session.map([Query(tail="Which titles are eligible?", output=Answer, id="titles")])
        .check(CitationCheck(), repair=CitationRepair(max_rounds=2))
        .filter(lambda item: bool(item.quote))
        .reduce(output=FindingList, instructions="Merge the cited facts.")
        .run()
    )
    print(result.failures, result.cache_hit_rate)


asyncio.run(main())
```

`examples/run_offline.py` is a fuller script: a `FunctionModel` walks map, citation check, a coverage policy, and gap-fill against the same notice.

## Why

A long document plus a short question is a bad unit of work when you have dozens of questions. Sending the document on every call spends the same input tokens again, and the answers do not share a schema or a way to say where they came from.

Prefix caching changes the cost shape. Providers that cache a stable prompt prefix charge much less, and answer much faster, for the tokens they have already seen. That only works when the prefix is byte-for-byte stable: same tools, same instructions, same documents, and a tail that varies. forkreduce's job is to keep that prefix frozen and to give the map/reduce steps around it a place to live.

The framework does not know your domain. It will not decide that a missing "deadline" is fatal or that a citation failure should be dropped. It reports the facts and runs the action your policy returns.

## Concepts

| Piece | What it is |
| --- | --- |
| **Session** | Built by `build()`. Holds the frozen prefix, the output types, the model, and the execution policy. |
| **Prefix** | Instructions + cleaned documents + the end marker, and (in tools mode) the tool set. Hashed for cache identity. |
| **Query** | A tail, an expected output type, and optional message history for continuing a branch. |
| **Map** | Many queries on one session. The first batch can warm the cache before concurrency opens. |
| **Check / repair** | A check reports problems. A repair may continue the branch, and code copies back only the allowed fields. |
| **Filter** | Your predicate. Outputs that fail it stay on the run and are left out of the reduce. |
| **Reduce** | One or more queries on the same prefix. Entries carry `source_ids`. A coverage policy decides what to do with gaps. |
| **Follow-up** | A hook between rounds. The built-in `GapFill` asks for missing leads and turns them into the next map. |

### Query and reduce flow

```mermaid
flowchart TB
    context["Shared context<br/>Tools · instructions · documents"]

    subgraph map["Map · parallel queries"]
        direction LR
        q1["Question A"]
        q2["Question B"]
        q3["Question C"]
    end

    context --> q1
    context --> q2
    context --> q3

    q1 --> check["Check and bounded repair · optional"]
    q2 --> check
    q3 --> check
    check --> filter["Business filter · optional"]
    filter --> reduce["Reduce · merge results, retain provenance"]
    reduce --> result["Results and reports<br/>Covered · uncovered · failures"]

    result -.->|"Follow-up · optional: generate new questions"| map

    style check stroke-dasharray: 5 5
    style filter stroke-dasharray: 5 5
```

### What sits in the cache prefix

```mermaid
flowchart LR
    subgraph prefix["Frozen prefix (hashed)"]
        tools["Output tool set<br/>name, description, JSON schema"]
        instructions["Instructions"]
        docs["Cleaned documents"]
        marker["===END_FIXED_DOCUMENTS==="]
        tools --> instructions --> docs --> marker
    end
    tail["Query tail<br/>+ repair / reduce / gap text"]
    prefix --> tail
    tail --> model["Model call"]
```

In `prompted` mode the tool list is empty. The expected schema is appended to the tail, so it is not part of the prefix.

### Tool set

```mermaid
flowchart TB
    session["One Agent, one tool set, registered at build()"]
    session --> answer["Answer"]
    session --> leads["LeadList"]
    session --> findings["FindingList"]
    session --> reasons["DiscardExplanations"]
    q1["Query expects Answer"] --> session
    q2["Reduce expects FindingList"] --> session
    q3["RequireReason expects DiscardExplanations"] --> session
```

Every output type you will ask for, including `DiscardExplanations` if a policy may request reasons, is registered up front. A later call that names an unregistered type raises `UnregisteredOutputError` before any model call. In tools mode, a reply that uses a different registered tool is a validation failure and is retried.

## API by concept

### Session and `build`

```python
session = build(
    model,                          # any pydantic-ai Model
    documents=notice,               # str, sequence of str, Block, Document, or a sequence of those
    instructions="...",
    outputs=[Answer, FindingList],  # unique BaseModel subclasses; tool name is the class name
    output_mode="auto",             # default; or explicitly "tools" / "prompted"
    policy=Policy(concurrency=8, retries=2),
    warmup=True,
    executor=None,                  # default LocalExecutor(policy)
    model_settings={"temperature": 0},  # sent on every call
)
```

`session.prefix_hash` is a SHA-256 of a canonical JSON document: version, **resolved** output mode, instructions, cleaned documents, and each tool's name, description, and schema. Rebuilding the same session produces the same hash. Changing a tool schema, the instructions, or the document text changes the hash. The hash algorithm is unchanged: default `"auto"` resolving to tools preserves the old tools hash; resolving to prompted changes the old default hash. Explicit tools/prompted hashes remain stable. There is no `"auto"` marker in the hash.

`build()` defaults to `output_mode="auto"`: it chooses tools only when the resolved model profile and effective settings support **forced output-tool use**, otherwise prompted JSON validated against the same registered Pydantic type. `session.output_mode` retains the requested mode (`"auto"`, `"tools"`, or `"prompted"`); `session.resolved_output_mode` is `"tools"` or `"prompted"`. Every actual `QueryResult`, including failures and combined reduce results, records `resolved_output_mode`; `None` is only the legacy/manual-construction default. This is a build-time capability decision, not a fallback after a failed provider request. Tools mode never silently accepts JSON text or switches modes after a response. Registered schemas remain part of `prefix_hash` in prompted mode, although their JSON is sent only in query tails, not in the provider's shared prompt prefix.

Explicit modes are honored. Explicit `"tools"` without effective forced-tool support emits `UnforcedToolsWarning` by default; `on_unforced_tools="error"` raises `UnforcedToolsError`, and `"ignore"` suppresses this diagnostic (neither setting adds provider capability). The policy applies at build and when query settings change capability. For an auto session, a `Query.model_settings` override that would change the resolved mode raises `OutputModeConflictError` before budget reservation or a provider call: build a new session with those settings instead. Settings merge shallowly in order: model defaults, build settings, query overrides; a query `extra_body` replaces the whole session value.

The limitation can belong to the **model/vendor API**, especially in thinking mode, rather than a gateway. [MiMo's OpenAI-compatible API reference](https://mimo.mi.com/docs/en-US/api/chat/openai-api) documents that non-`auto` `tool_choice` values are removed and behave as `auto`; [DeepSeek issue #1376](https://github.com/deepseek-ai/DeepSeek-V3/issues/1376) reports HTTP 400 for forced choices while thinking is enabled. [Pydantic AI PR #7450](https://github.com/pydantic/pydantic-ai/pull/7450) describes request-dependent forcing and implicit downgrades to `auto`. These sources do not establish that every model or endpoint has the same restriction. Supply an accurate `profile` for a custom endpoint; `supports_tools=True` alone does not promise forced output.

In installed Pydantic AI 2.61, the public `ModelProfile` fields all exist with these defaults:

| Field | Default | Meaning |
| --- | --- | --- |
| `supports_tools` | `True` | Can use tools at all. |
| `supports_forced_tool_choice` | `True` | Accepts required/named forced tool choice. When false, SDK-generated forcing can become `auto`; explicit forcing raises an SDK `UserError`. |
| `supports_forced_tool_choice_with_thinking` | `True` | Forcing remains supported while this request thinks. |
| `thinking_enabled_by_default` | `False` | Thinks when no setting overrides it; does not mean thinking is impossible to disable. |
| `supports_thinking` | `False` | Supports unified thinking configuration; when false the SDK ignores unified `thinking`. |
| `thinking_always_enabled` | `False` | Thinking cannot be disabled; implies thinking support, and `thinking=False` only selects the lowest accepted effort. |
| `forced_tool_choice_disables_thinking` | `False` | Forcing would suppress thinking; while thinking is active, the SDK avoids its implicit output-tool forcing. |

`default_structured_output_mode="tool"` is the SDK's preference, not a capability guarantee. forkreduce's auto rule selects only tools or prompted output, never native structured output.

The exact rule requires both tools and forced-tool support; while thinking is active it also requires `supports_forced_tool_choice_with_thinking=True` and `forced_tool_choice_disables_thinking=False`. `thinking_always_enabled` always wins. Recognized provider controls take precedence over unified `thinking`: `openai_reasoning_effort` / `extra_body.reasoning_effort` (`none` disables); `anthropic_thinking.type` / `extra_body.thinking.type` (enabled/adaptive enables, disabled disables); boolean `extra_body.chat_template_kwargs.enable_thinking` (also in top-level `chat_template_kwargs`); and `google_thinking_config.thinking_budget` (zero disables, other values including -1 enable) or `thinking_level` (minimal/low/medium/high all enable). Conflicting controls conservatively mean active thinking. Otherwise unified `thinking` is honored only with `supports_thinking=True`, then the rule falls back to `thinking_enabled_by_default`. Unknown non-`None` thinking/reasoning-shaped settings, including malformed recognized controls, raise `ForkReduceError` with code `unknown_thinking_control` in auto; explicit tools treats forcing as unavailable and applies the warning policy, while explicit prompted is honored. Display-only `include_thoughts`, `openai_reasoning_summary`, and `openai_reasoning_generate_summary` do not enable thinking. There are no model-name guesses.

Thinking controls are inspected only when they can change forcing capability: if tools or forcing are unsupported, auto resolves directly to prompted; if forcing works with thinking and does not disable it, auto resolves directly to tools. These unconditional profiles do not reject irrelevant unknown thinking controls. The unknown-control error and conservative explicit-tools diagnostic above apply only to conditional profiles.

Give reasoning models output headroom: a provider's completion cap can include both reasoning tokens and the visible answer. A small `max_tokens` may be exhausted before any JSON or tool arguments appear. For example, pass `model_settings={"max_tokens": 8192}` to `build()` (or `max_tokens=8192` to `openai_compatible_model()`) and size it for your workload and provider; 8,192 is an example, not a guarantee against truncation or a billing limit. Do not assume a gateway supports reasoning-effort controls without checking its contract.

[MiMo's deep-thinking guide](https://mimo.mi.com/docs/en-US/quick-start/usage-guide/text-generation/deep-thinking) also documents that its listed thinking models ignore requested `temperature`/`top_p` and use `1.0`/`0.95`. Thus requesting `temperature=0` does not guarantee deterministic output in that mode.

`documents` accepts a string (one block, id `b1`), a sequence of strings (`b1`, `b2`, ...), or `Block` / `Document` values. Text is stripped of control characters before it is rendered or hashed. The rendered prefix ends with a line `===END_FIXED_DOCUMENTS===`.

`Block.meta` is stored and, by default, not rendered, so existing prefix hashes stay the same. Opt in with `meta_keys` (a list of keys, rendered in that order) or `meta_renderer` (a callable). Rendered meta is part of the prefix and the hash. Pass one of the two, not both.

```python
session = build(
    model,
    documents=Document(
        doc_id="notice",
        blocks=[Block(block_id="b1", text="Hours are 9 to 5.", meta={"page": "3"})],
    ),
    instructions="Answer from the notice.",
    outputs=[Answer],
    meta_keys=["page"],  # writes [meta {"page":"3"}] under the block
)
```

### Query and QueryResult

```python
query = Query(
    tail="Which titles are eligible?",
    output=Answer,          # must be registered
    id="titles",            # assigned q1, q2, ... when omitted
    history=None,           # None starts at the prefix; previous.messages continues a branch
    meta={},
    round=0,
    model_settings={"temperature": 0.2},  # overrides the session value for this call
)
result = await session.ask(query)
# or
results = await session.fork([query, other])
```

`QueryResult` fields:

| Field | Meaning |
| --- | --- |
| `query_id` | The query id. |
| `output` | The parsed model, or `None` on failure. |
| `messages` | Full branch transcript (`all_messages()`), so you can continue it. |
| `prefix_hash` | The session hash this call used. |
| `usage` | pydantic-ai `RunUsage` for this result, including retries. |
| `cache_hit_tokens` | `cache_read_tokens`, or `details["prompt_cache_hit_tokens"]` when the first is zero. |
| `attempts` | How many model calls this result took. |
| `failure` | A `Failure`, or `None`. |
| `round` | Plan round, starting at 1 inside `Plan.run`. |
| `elapsed_ms` | Wall time summed across attempts. |
| `needs_review` | Set when repair was exhausted and the policy says to keep the output flagged. |
| `dropped` | Set when the output is kept on the run but excluded from reduce. |
| `check_failures` | Problems the check still reports. |

Passing a `history` that does not start with the rendered documents emits `PrefixMismatchWarning` and still runs. The framework does not guess whether you meant a new branch.

`session.map(queries)` returns a lazy `Plan`. `session.reduce(...)` runs a reduce immediately and returns a `ReduceOutcome`.

### Check and repair

`CitationCheck` walks nested models and lists. For each `quote` / `block_id` pair (those field names are the default) it reports:

* `empty_quote` or `empty_block_id`
* `unknown_block` when the id is not in the session documents
* `quote_not_found` when the quote is not a substring of that block after whitespace is collapsed

Matching is case-sensitive.

`Citation` is a reusable Pydantic model with described `quote` and `block_id` fields: use it in a list or inherit it to add answer fields, as above. Each quote must be one contiguous passage entirely inside the single referenced block. Text appearing elsewhere in the document is not a valid quote for that block. Split evidence crossing block boundaries into separate citations with their own block IDs; do not concatenate blocks into one quote.

Append the opt-in `CITATION_INSTRUCTIONS` string to your session instructions when asking for citations. Nothing is injected into unrelated sessions, and the schema and instructions do not replace `CitationCheck`. Adopting `Citation` changes the registered output schema and hence `prefix_hash`; appending the instructions also changes the frozen prompt and hash. Existing custom schemas and instructions retain their previous hashes. Reduce and repair guidance changes only query tails, not the frozen prefix or hash.

`CitationRepair(max_rounds=2, allowed_fields=("quote", "block_id"))` continues the same branch and asks for the same output type. Its instructions repeat the single-block rule; under the default field restriction, repair must shorten a quote or correct its block ID rather than add or split output entries. Whatever the model returns, `apply` deep-copies the original object and copies only the allowed fields at each failing path. Other fields stay as they were. There is no separate repair schema, so the tool set does not change.

```python
session.map(queries).check(
    CitationCheck(),
    repair=CitationRepair(max_rounds=2),
    on_failure="needs_review",  # or "drop" or "fail"
)
```

`on_failure` overrides `Policy.repair_failure` for items that still fail when repair stops:

* `needs_review` keeps the output and flags `needs_review`
* `drop` excludes it from the reduce and leaves it on the run
* `fail` records a permanent failure, skips the reduce, and does not start later rounds

The default is `needs_review`.

A custom check is a callable `(output, *, documents) -> list[CheckFailure]`. A custom repair implements `max_rounds`, `build_query`, and `apply`.

### Filter

```python
.filter(lambda item: bool(item.quote))
```

Predicates are AND-ed. A rejected output remains on `RunResult.results` and is not passed to the reduce. The framework ships no deterministic reducer; filtering is your hook.

### Semantic reduce and coverage

A reduce is a normal query on the same session, so it hits the same prefix. The tail lists each input as `id: ...` plus its JSON and an explicit allowed source-ID list. `source_ids` must identify those input items (for map results, the query/fork IDs), not document or block IDs found inside their values. Replace inherited `source_ids` with the current batch's input IDs; do not mix in evidence addresses.

When inputs contain string `question_id` fields, including nested models, dictionaries, lists, and tuples, the tail also maps each question to its allowed input IDs, for example `{"opening": ["fork:0"], "closing": ["fork:1"]}`. Repeated questions collect all matching input IDs. This is generic guidance, not a domain-specific validator; the provenance check below remains unchanged. Inputs without question IDs work as before, and tree batches use their current alias IDs before final expansion.

```python
class ExplainUncovered:
    def decide(self, report):
        if report.uncovered:
            return RequireReason(tuple(report.uncovered))
        return Accept()

session.map(queries).reduce(
    output=FindingList,                 # registered, and entries must carry source_ids
    instructions="Merge the notice into one list.",
    coverage=ExplainUncovered(),
    max_batch=200,                      # larger inputs become a tree of reduces
    max_rounds=3,                       # bound on coverage Retry
)
```

`Provenance` (the default check) requires every traced entry to cite ids that exist. Unknown ids land in `invalid_refs`. Entries with no `source_ids` land in `untraced`. Both are retried inside the batch, up to `Policy.retries`, with a hint to cite only the given ids. A failed model call is not asked again here; the executor already retried it.

`ReduceReport` always lists:

| Field | Meaning |
| --- | --- |
| `covered` | Input ids that appear in some `source_ids`. |
| `uncovered` | Input ids that do not. Present even when the policy accepts. |
| `invalid_refs` | Cited ids that are not in the input. |
| `untraced` | JSON paths of entries that declare no source ids. |
| `reasons` | `DiscardExplanation` rows after `RequireReason`. |
| `kind` | `"reduce"` by default. |

`CoveragePolicy.decide(report)` returns an action. The framework executes it:

| Action | Effect |
| --- | --- |
| `Accept()` | Keep the merge. Uncovered ids stay on the report. This is the default (`AcceptUncovered`). |
| `RequireReason(items)` | Continue the reduce branch and ask for `DiscardExplanations`. The type must already be registered. |
| `Retry(items, hint)` | Reduce again. An empty `items` retries every original id; otherwise only those ids. `hint` is appended to the tail. |
| `Fail(reason)` | Stop. `ReduceOutcome.fatal` is set and the reason is stored. |

`Drop` is not a reduce action. It is used by citation-failure handling and by gap-fill lead policies.

More than `max_batch` inputs are reduced in a tree. Intermediate nodes get alias ids. Before the coverage policy runs, those aliases are expanded back to the original ids on the returned objects. The message history still contains the model's raw tool arguments, which may show the alias ids.

`Traced` is a pydantic base with `source_ids: list[str]`. Subclass it, or declare your own `source_ids` field. If a subclass redeclares the field, give it a default so the model can omit it only when you mean that; the provenance check still treats a missing or empty list as untraced.

### Follow-up and gap-fill

```python
def more(state, session):
    if state.round >= 2:
        return []
    return [Query(tail="What is still missing?", output=Answer)]

session.map(queries).reduce(output=FindingList).followup(more, max_rounds=2)
```

`followup(fn, max_rounds=2)` calls `fn(state, session)` after each round except the last. `max_rounds` counts map rounds, including the first. An empty list ends the run. Without `followup`, a plan runs a single round.

`RoundState` carries `round`, `results`, `outputs`, `merged`, `report`, plus `failures` and `side_results` for calls the hook itself makes. Items that declare a `round` field are stamped with the map round when the round index is greater than zero.

`GapFill` is a built-in strategy:

```python
gap = GapFill(
    lead_output=LeadList,          # registered; reads .leads or .items
    item_output=Answer,            # registered; one query per new lead
    scopes=[Scope("delivery", "how a sealed offer is delivered")],
    instructions="Report blocks in this scope that the merge does not already cite.",
)
```

It asks once per scope, on the same prefix. Leads whose position (`block_id` by default) is already present on the merged output are skipped. A `LeadPolicy` then sees a `LeadReport` and may `Accept`, `Drop` specific positions, or `Fail`. The default is `AcceptLeads`.

### Concurrency, retries, budget

```python
Policy(
    concurrency=8,             # semaphore around model calls
    retries=2,                 # extra attempts after the first
    retry_backoff=0.5,         # seconds; doubles each transport retry, capped by max_backoff
    max_backoff=8.0,
    budget=Budget(
        max_requests=100,
        max_input_tokens=None,
        max_output_tokens=None,
        max_total_tokens=None,
    ),
    repair_failure="needs_review",
    budget_policy=None,        # callable(BudgetReport) -> Accept | Fail | ...
    context_limit=None,        # per-call window in estimated tokens; None skips the local warning
    context_reserve=None,      # prefix headroom; None means min(limit, max(4096, limit // 4))
)
```

Failure classes:

| Kind | Examples | What happens |
| --- | --- | --- |
| `validation` | Wrong tool, schema mismatch, `UnexpectedModelBehavior`, `UsageLimitExceeded` | Retried immediately, up to the bound. |
| `transport` | HTTP 408, 409, 425, 429, and 5xx; timeouts; connection errors | Retried with exponential backoff. |
| `permanent` | Other 4xx, `UserError`, provider `context_overflow`, anything unrecognized | Not retried. Returned on the result. |
| `budget` | A ceiling was already crossed | Not retried. |

Each model call is built with pydantic-ai `retries=0` and `UsageLimits(request_limit=1)`. forkreduce owns the retry loop, so a validation error does not become a hidden inner retry. Failed calls are never removed from the run.

`warmup=True` (the default) runs the first query of the first `fork` / `map` batch to completion before starting the rest, so the provider can store the prefix. There is no extra dummy request. Later forks on that session run fully concurrent. `ask` does not warm up by itself.

Token budgets are cumulative. When any token ceiling is set, model calls are serialized even if `concurrency` is greater than 1. A call is refused only when completed calls have already reached the ceiling. The one in-flight call that crosses a ceiling is allowed, because its size is known only after it returns, and nothing else starts until that usage is recorded. Overshoot is at most that one call, not one call per concurrent slot. Request-count budgets stay concurrent: each call reserves one request before it is sent. A call blocked before it is sent reports `attempts=0` and does not reserve a request.

A response that fails validation is still billed. Its token usage is read from the captured model response and added to the budget and to `QueryResult.usage`. A transport failure that never produces a response has no tokens to recover; the reserved request still counts. `budget_policy` may return `Accept` to force a blocked call through, or `Fail` to block with your reason. The default blocks.

### Context limit

`Policy.context_limit` is the model context window in estimated tokens, separate from the cumulative budget above. The estimate is one token per four characters, rounded up, over the instructions, the tool schemas, the documents (or the continuation history), and the query tail. It is not the provider's tokenizer. The provider is the source of truth: a local estimate never blocks `build()`, `ask`, `fork`, or `map`, and the prompt is not truncated or split into chunks.

When a limit is set and the estimate is over the usable window, forkreduce emits `ContextEstimateWarning` (a `UserWarning`) and logs the same message on the `forkreduce.context` logger, then sends the request anyway.

* At `build()`, the frozen prefix warns when `prefix_estimate > context_limit - reserve`. The warning is stored on `Session.context_warning`.
* At `ask` / `fork` / `map`, a query warns when the full estimate (prefix or history, plus the tail) is greater than `context_limit`. The warning is stored on `QueryResult.context_warning` and listed on `RunResult.context_warnings`. The prefix warning is also copied onto `RunResult.context_warning`.

Each warning records `estimated_tokens`, `limit`, and `reserve`. Equality is inside the window. `context_limit=None` skips the local warning even if `context_reserve` is set.

`context_reserve` is prefix headroom, in the same estimated tokens. `None` (the default) means `min(context_limit, max(4096, context_limit // 4))`. 4096 tokens leaves room for a real tail and a structured reply on an 8k–16k window (half of 8k, a quarter of 16k). On a larger window, a quarter of `context_limit` is the larger of the two, so a prefix that crowds out the per-query work still warns. The value is capped at `context_limit`, so a window smaller than 4096 treats the whole window as headroom and warns on any non-empty prefix estimate without going negative. Set an integer, including `0`, to replace the formula. `0` warns on the prefix only when the prefix itself exceeds the full window. The reserve is not subtracted again for a request warning, because that estimate already includes the tail; it is still recorded on the warning.

A provider that rejects the prompt for context length (HTTP 400 whose body says the maximum context length was exceeded) is a non-retryable `Failure` with code `context_overflow` and `detail["where"] == "provider"`. When the body includes the figures, the message reports those (`reported 9000 tokens exceeds context limit 8192`). Chunked map-then-reduce is not part of v0.1.

`RunResult.usage` is a `UsageSummary`: request count, input and output tokens, cache-hit tokens, elapsed milliseconds, a per-round breakdown, and `cache_hit_rate` (`cache_hit_tokens / input_tokens`). Map calls and their repairs are folded onto the map `QueryResult` so tokens are counted once. Reduce calls are added from the individual calls, not from a second summed copy.

### Continuing a branch

```python
first = await session.ask(Query(tail="Which titles?", output=Answer, id="titles"))
follow = await session.ask(Query(
    tail="Now quote the deadline as well.",
    output=Answer,
    id="titles-more",
    history=first.messages,
))
```

`history` is the full branch, prefix included. Repair and `RequireReason` use the same mechanism.

### Models

Pass any pydantic-ai `Model`, including `TestModel` and `FunctionModel`. For an OpenAI-compatible server:

```python
from forkreduce import openai_compatible_model

model = openai_compatible_model()  # OPENAI_MODEL or FORKREDUCE_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

model = openai_compatible_model(
    "local-model",
    base_url="http://127.0.0.1:8000/v1",
    timeout=30,
    temperature=0,
    profile={"supports_tools": True},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

The OpenAI client is constructed with `max_retries=0`. forkreduce does the retrying. `timeout`, `profile`, `extra_body`, and the usual sampling keys are pydantic-ai model settings. A `settings={...}` dict passes any other `ModelSettings` key; an explicit argument overrides the same key. `build(..., model_settings=...)` applies settings to every call, and `Query.model_settings` overrides keys for that call. `extra_body` is replaced as a whole, not merged field by field.

### Temporal (optional)

Install `forkreduce[temporal]`; core imports never load `temporalio`. Each `forkreduce.run_query` activity calls `Session.run_once` exactly once. Temporal owns retries: validation/transport failures remain retryable, permanent/budget failures do not. Defaults are unchanged: one `forkreduce.map_queries` workflow, five-minute start-to-close, no heartbeat timeout, and `Policy.retries + 1` total attempts.

#### Options and plugin worker

`ActivityOptions` durations are **seconds**. Set them on `build`, override one payload's `activity_options`, or replace them for the whole workflow via `MapQueriesInput.activity_options`. Unspecified retry fields inherit the payload's policy-derived budget/backoff; `maximum_attempts` includes the first attempt. Recipe: **2 attempts / 3 minutes / 30-second heartbeat**.

```python
from forkreduce import ActivityOptions, build
from forkreduce.contrib.temporal import (
    MapQueriesInput, MapQueriesWorkflow, QueryBatchWorkflow,
    query_activity_input, register_session, run_query_activity,
)
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
from temporalio.client import Client
from temporalio.worker import Worker

session = build(
    model, documents=document, instructions="Answer from the document.", outputs=[Answer],
    activity_options=ActivityOptions(
        start_to_close=180, heartbeat_timeout=30, maximum_attempts=2,
        # schedule_to_close=600, retry_backoff=1, max_backoff=8, backoff_coefficient=2,
    ),
)
register_session(session)  # every activity worker process
client = await Client.connect("localhost:7233", plugins=[PydanticAIPlugin()])
async with Worker(
    client, task_queue="extraction",
    workflows=[MapQueriesWorkflow, QueryBatchWorkflow],
    activities=[run_query_activity],  # inherits the client's plugin
):
    results = await client.execute_workflow(
        MapQueriesWorkflow.run,
        [query_activity_input(session, query_id="q1", tail="What is required?", output_name="Answer")],
        id="extraction-run-1", task_queue="extraction",
    )
```

Both Temporal's default converter and `PydanticAIPlugin`'s `PydanticAIPayloadConverter` have offline workflow/replay coverage. forkreduce's per-query activities coexist with official `TemporalDurability` agents on the same worker: register those agents/workflows normally, without wrapping these activity calls in another retry loop. Install Pydantic AI's `[temporal]` extra for its official integration. For a default-converter worker, set `workflow_runner=SandboxedWorkflowRunner(restrictions=SandboxRestrictions.default.with_passthrough_modules("forkreduce", "pydantic_ai"))`; both classes come from `temporalio.worker.workflow_sandbox`. This avoids reimporting model dependencies inside the sandbox.

When configured, periodic heartbeats continue while awaiting the model. They signal liveness, not business progress. Blocking the event loop prevents heartbeats: move CPU work to a thread/separate activity and keep a finite start-to-close timeout. Local execution ignores `ActivityOptions`; they never change prefix bytes/hash.

Registration is process-local. Payloads carry hashes, tails, output names, options, and optional per-query `model_settings`, never model instances or Python classes. Conflicting model/request-body settings/profile/policy registrations raise `ForkReduceError`; unregister before replacement. `timeout`/`extra_headers` remain outside the fingerprint. Settings must be JSON-serializable. `activity_id(prefix_hash, query_id)` remains `forkreduce:{prefix_hash}:{query_id}`. Inline single-workflow continuations still accept the previous `messages_json` as `history_json`.

#### Bounded batches and references

`MapQueriesInput(queries=payloads, batch_size=15, child_concurrency=4)` runs bounded child workflows; register **both** `MapQueriesWorkflow` and `QueryBatchWorkflow`. Results preserve input order. Child IDs derive from the parent workflow ID, prefix hash, and batch index. They stay stable on replay and workflow retries that change the run ID; distinct parent workflow IDs do not collide. This is orchestration identity, not exactly-once provider billing.

Batch mode requires a `MessageStore` and returns `messages_ref` plus `output_ref`, not transcript/structured-output bodies. Parent history is independent of transcript size, but query tails, metadata, and reference lists still occupy history. Keep them small. Batch continuations require `history_ref`, not inline `history_json`. Single-workflow/list-input mode keeps inline behavior for small cases.

`MessageStore` and `PrefixStore` use async `put(key, bytes) -> ref` / `get(ref) -> bytes`. Implement durable object storage/database access shared by every worker; `InMemoryMessageStore` is process-local, for examples/tests only. Freeze large prefixes outside workflow code, e.g. in an application's `prepare` activity, returning only references and small query payloads:

```python
from forkreduce.contrib.temporal import (
    freeze_prefix, register_session_factory, set_prefix_store, set_message_store,
)

# Client or prepare activity; store is your durable implementation.
prefix_ref = await freeze_prefix(session, store)
payloads = [
    query_activity_input(
        session, query_id=f"q{i}", tail=question, output_name="Answer", prefix_ref=prefix_ref,
    )
    for i, question in enumerate(questions)
]

# Every worker: model credentials and output classes stay local.
def restore(frozen):
    return build(
        model, documents=frozen.documents, instructions=frozen.instructions,
        outputs=[Answer], output_mode=frozen.output_mode,
        on_unforced_tools=frozen.on_unforced_tools,
    )

set_prefix_store(store)
set_message_store(store)
register_session_factory(session.prefix_hash, restore)
# Execute while the worker above is running:
results = await client.execute_workflow(
    MapQueriesWorkflow.run,
    MapQueriesInput(queries=payloads, batch_size=15, child_concurrency=4),
    id="extraction-run-2", task_queue="extraction",
)
answer = Answer.model_validate_json(await store.get(results[0].output_ref))
# Continue with history_ref=results[0].messages_ref.
```

`FrozenPrefix` stores documents, instructions, rendered prefix and tool identity, not executable model/schema objects. Reconstruction verifies the prefix hash: use matching schemas/rendering configuration. Configure factories at worker startup and retain blobs through retry/continuation lifetimes. Persist references and final application results; completed workflow streams are not a replayable archive. Drain in-flight workflows or use worker versioning/new task queues for incompatible orchestration changes, and replay representative old histories before migrating. Converter tests are not a guarantee that arbitrary old workflow code replays after orchestration changes.
`FrozenPrefix.output_mode` retains the requested mode, including `"auto"`; `FrozenPrefix.resolved_output_mode` records the actual mode (`None` for legacy snapshots), and `on_unforced_tools` preserves the diagnostic policy. Restore the requested mode and policy as above, and supply the same thinking-related model/session settings in the factory. Workers verify the reconstructed resolved mode and prefix hash: changed capabilities/settings must fail rather than silently restore a different output contract. The factory still owns model/settings; settings that may contain transport credentials are not serialized into the snapshot.
Factories run per activity attempt by default; this avoids silently sharing mutable session budgets across workflows. For worker-local reuse, the factory may return an explicitly managed cached session or call `register_session`. Applications own that cache's synchronization, lifetime, and budget scope.


## Caching caveats

**Fixed schema.** Tool definitions are part of the prefix. Providers render tools before messages, and forkreduce does not add tools later. A new output type, a renamed field, or a changed description changes `prefix_hash` and misses the cache for every query, not just the one with the new tail. `Literal` enums need an `"other"` (or similar) escape so one unexpected value does not force a schema change later.

**Control characters.** Some providers reject tool-call JSON that contains raw control characters. forkreduce strips C0 controls other than tab, newline, and carriage return, and strips DEL, before rendering and hashing. The hash of a document with a null byte matches the hash of that document without it.

**Warmup.** The first request creates the cache entry. `warmup=True` finishes that request before the rest of the first batch starts. Firing the whole batch at once can miss the cache on every call.

**Cache expiry.** Providers expire prefix caches on their own schedule (often on the order of minutes of idle time, sometimes longer). forkreduce cannot refresh a cache it does not own. A later batch after a long gap may miss, pay full input price, and then hit again. The hash still tells you whether you sent the same prefix.

**History.** Continuing a branch sends the prior question and answer as well as the prefix. That also hits a prefix cache, including the earlier turns, as long as those bytes are unchanged. Timestamps and run ids are not part of the prefix check; the check compares the rendered document text.

**Prompted mode.** A session resolved to `"prompted"` puts no tools on the agent. The schema of the expected type is appended to the tail only, so different queries can use different registered types without busting the prefix. Tools mode keeps one stable tool list when forced output-tool use is supported; default `"auto"` selects it only in that case.

**Sampling and transport.** `temperature`, `top_p`, `max_tokens`, `seed`, penalties, `stop_sequences`, `timeout`, and `extra_headers` do not change the frozen prefix, so they do not bust a provider prefix cache. `extra_body` does not change those bytes either, except that thinking settings can change auto's resolved output mode when building a session (and therefore its hash); query overrides cannot flip that mode. Sampling parameters and `extra_body` do change the JSON body, so Temporal treats a different value as a different execution and `register_session` rejects it. `timeout` and `extra_headers` are transport options and are left out of that fingerprint.

**Profile and block meta.** A model `profile` can change how tools or schemas are rendered on the wire. That can miss the provider cache even when `prefix_hash` is unchanged, which is why the resolved profile is part of the execution fingerprint. `Block.meta` is omitted unless you pass `meta_keys` or `meta_renderer`. Opting in writes meta into the document text, which changes `prefix_hash` and the cached prefix.

## FAQ

**Does the reduce have to use the same output type as the map?** No. It must be one of the types registered at `build()`. Map queries and the reduce can differ.

**What if the model cites an id I never sent?** It is `invalid_refs`. The batch retries up to the policy bound. If the model keeps inventing ids, they stay on the report and the default policy still accepts the merge. A stricter policy can `Fail` or `Retry`.

**Are uncovered items deleted?** No. `Accept` keeps them on `ReduceReport.uncovered`. Use `RequireReason` when a person should see why they were left out.

**Can I reduce without a plan?** Yes. `await session.reduce(items, output=FindingList, id_of=...)`.

**Where do API keys live?** Only in the environment, or in arguments you pass to `openai_compatible_model`. This repository has no `.env` and no embedded keys.

**Does core import Temporal?** No. A subprocess that imports `forkreduce` does not load `temporalio`.

## Development and testing

From a checkout:

```bash
pip install -e ".[dev]"
ruff check src tests examples && ruff format --check src tests examples
mypy
pytest
```

The 0.2.0 development checks cover Pydantic AI 2.61.0 / OpenAI 3.37.0 with Temporal SDK 1.33.0 and 1.34.0. Requirements remain `pydantic-ai>=2.57,<3` and optional `temporalio>=1.9`; the official plugin has its own extra requirements. Temporal tests start a real time-skipping server and use only offline models. Set `TEMPORAL_TEST_SERVER=/path/to/temporal-test-server` to reuse an existing binary in network-isolated environments; otherwise the SDK downloads it.

Examples run offline with no key:

```bash
python examples/run_offline.py
```

A live OpenAI-compatible endpoint is optional. The process exits immediately when `OPENAI_API_KEY` is unset.

```bash
OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... python examples/run_live.py
```

`FORKREDUCE_MODEL` is accepted as another name for the model.

## Releasing

PyPI publishes from a git tag. `.github/workflows/release.yml` builds the sdist and wheel, checks them, and uploads with [Trusted Publishing](https://docs.pypi.org/trusted-publishers/). The repository has no PyPI token. The publisher is the `pypi` environment on `426ai/forkreduce`, workflow `release.yml`.

1. Set the same version in `pyproject.toml` (`version`) and `src/forkreduce/__init__.py` (`__version__`).
2. Commit that change on `main`.
3. Tag that commit and push the tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The tag is `v` plus the version string: `0.1.0` is `v0.1.0`. The workflow fails when they differ. You can also run the Release workflow by hand from the Actions tab, but only on an existing `v*` tag.

## Roadmap

v0.1.0 is the first public cut: one session, local execution, citation check/repair, semantic reduce with coverage actions, gap-fill, usage stats, and an optional Temporal activity wrapper.

Not in this version, and reasonable later:

* more built-in checks than citations
* a durable `MessageStore` implementation (object storage, a database)
* streaming partial results
* explicit cache-warmup metrics separated from the first real query
* a second workflow runtime behind the same `Executor` protocol
* opt-in chunked map-then-reduce when a prompt is larger than the context window (the local estimate only warns; a provider context-length error still fails the call)

## License

MIT. See [LICENSE](https://github.com/426ai/forkreduce/blob/main/LICENSE).
