# CUAD multi-question benchmark

Source-tree-only evaluation of **one long contract + N questions**, not a claim that
forkreduce wins. No dataset, provider responses, or credentials are checked in.
The package version and core dependencies are unchanged; `benchmarks/` is excluded
from the wheel by the existing explicit `src/forkreduce` wheel package selection.

## Install and run without paying

From the repository root, Python 3.11+:

```sh
python -m pip install -e '.[dev]' -r benchmarks/requirements.txt
python -m benchmarks smoke
python -m benchmarks estimate
# Equivalent offline estimation path:
python -m benchmarks run --dry-run
python -m pytest
ruff check src tests examples benchmarks
ruff format --check src tests examples benchmarks
mypy
```

`smoke` uses a local Pydantic AI `FunctionModel`, exercises all three real adapter
paths (including search/read tools), scores a synthetic answerable and absent
clause, and creates temporary Markdown/CSV/SVG reports. Synthetic model usage is
only an accounting check, **not measured CUAD/provider performance**. CI installs
only the additional tokenizer and runs offline tests/smoke. Neither smoke nor
estimate constructs a live provider, even if API credentials exist in the environment.

`estimate` downloads the public CUAD archive and the local tokenizer vocabulary,
then counts actual contract/question tokens with `cl100k_base`. These downloads
are free and are not inference. Data lives in ignored `.cache/cuad/`; estimate
artifacts live in ignored `benchmarks/results/estimate/estimate.{json,md}`. The JSON
contains every contract × method × N workload, prices, assumptions, length
statistics, hashes, and a reproducible workload snapshot. The command prints the
artifact SHA256 for later approval. `--contracts`, `--seed`, `--n`, `--cache-dir`,
`--prices`, and `--output-dir` are configurable. Defaults are 20 contracts,
seed **20261024**, and N = **5, 20, 41**: **180 logical runs**. Price/cache
sensitivity cases are alternative estimates of these runs, not extra paid runs.

## Data, attribution, and limitations

- **CUAD v1**, The Atticus Project: 510 commercial contracts, 41 clause categories.
  [Official repository](https://github.com/TheAtticusProject/cuad),
  [project page](https://www.atticusprojectai.org/cuad),
  [dataset card](https://huggingface.co/datasets/theatticusproject/cuad-qa).
- Hendrycks, Burns, Chen, and Ball (2021),
  [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/abs/2103.06268).
- Dataset/annotations: **CC BY 4.0**; retain attribution and identify transformations.
  [License text](https://creativecommons.org/licenses/by/4.0/).
  The official dataset card explicitly makes **no representation or warranty about
  the underlying contracts' license status**. Contracts originated in public SEC
  EDGAR filings; public availability is not a blanket copyright or privacy waiver.
  Respect the project's [disclaimer](https://www.atticusprojectai.org/disclaimer)
  and [privacy policy](https://www.atticusprojectai.org/privacy-policy).
- Loader pins the official public release source and verifies archive/JSON SHA256.
  Original text and annotated offsets are preserved. Repeated category records
  are grouped; all distinct annotated spans are positive labels, not alternative
  interchangeable answers. The downloaded data is never copied into tests.
- Sampling is at the **contract level**, not QA-row level. Contract IDs are sorted
  before seeded selection; question-category order is seeded and shared across
  contracts. N=5 is a prefix of N=20, which is a prefix of N=41.
- This is an exploratory sample from the full release, **not a held-out leaderboard
  score**. CUAD may have appeared in model training data. Results on 20 contracts
  have limited power, and 41 categories include many absent clauses. Report
  answerable and absent metrics separately. Contracts/annotations contain PDF
  conversion artifacts, redactions, and sometimes nonliteral `<omitted>` spans;
  these are tracked, not silently rewritten into favorable gold.

## Methods and fairness contract

Every method receives the same model instance, temperature (default 0), common
`INSTRUCTIONS`, question template, and `Answers` JSON schema from `schema.py`.
The answer contains an ID and a list of verbatim citations; an empty list means
"no such clause". None receives gold answers. `source_ids` is an available schema
field for reduce provenance, not a separate richer answer format.

| Method | Access and execution | Budget / tradeoff |
| --- | --- | --- |
| `forkreduce` | Full contract frozen as shared prefix; one fork per question, framework citation check/repair, semantic reduce | Warmup enabled; concurrency 4; reduce/repair tokens are included, not hidden |
| `tool_agent` | One Pydantic AI agent, sequential questions in one retained conversation; lexical `search(query)` and `read_section(id, start, end)` over all overlapping chunks | Per question up to 8 requests and 12 tool calls; history grows and is billed again; actual position is recorded |
| `one_shot` | Full contract + all N questions in one structured request | No retrieval loss; long structured output/truncation risk is retained |

`MethodConfig` defaults: 6,000-character chunks, 600-character overlap, eight
search results, 24,000 characters per read, one schema retry, two citation-repair
rounds, maximum output 16,000 tokens per request. Tools are local, not paid
provider web-search tools. Read ranges allow inspecting relevant neighboring
text, rather than enforcing an artificially tiny retrieval window. All methods
have citation validation/repair opportunities; failed/missing/duplicate IDs and
post-reduce citation errors remain visible. The same chunk identifiers and
original text are used in retrieval and full-document paths.

Use `--method-config path.json` with JSON field overrides from `MethodConfig` to
change budgets (e.g. `{"requests_per_question": 12, "tool_calls_per_question": 20}`).
Use identical config when estimating and running. Estimator assumptions describe
**typical work plus contingency**, not a worst-case product of every maximum
budget; changing budgets requires reviewing assumptions and approving a new estimate.
Use `--estimate-config path.json` to override `EstimateConfig` planning assumptions,
for example `{"output_tokens_per_question": 512, "tool_loops_per_question": 3}`.
This changes estimates only, not actual tool budgets or model outputs.

Method adapters are registered in `methods.METHODS`; a callable accepts
`(contract, questions, model, config)` and returns `MethodResult`. Add future
per-question full-text/no-cache or RAG adapters without changing the metric schema.

Execution randomizes contract/N blocks and method order with the fixed seed to
reduce ordering effects. Provider caches cannot be reliably flushed, cache
sharing across runs is possible, and hits/TTL/thresholds are provider-dependent.
Warmup is an explicit framework strategy whose time/cost is counted. Do not
interpret a cost model's assumed hit rate as an observed hit rate. Compare methods
within a provider/model run; the two price profiles are sensitivity examples,
not a comparison between different model accuracies. Temperature 0 does not imply
bitwise determinism. A single full grid is not a repeated-run stability study;
schema failures/retries are the PoC stability measures.

## Metrics and artifacts

One JSONL record per contract × method × N contains model/config/version and
workload identifiers, answers, per-response provider usage, scored questions,
failures, retries, schema failures, concurrency, and **end-to-end wall time**.
Retries, tools, repair, reduce, and rejected structured responses are all billed
when the provider returns usage. No local tokenizer estimate is substituted for
observed usage. Missing usage remains unknown, not zero-cost. Cached input,
uncached input, cache writes, and output remain separate; cache writes are not
charged twice as ordinary input. Cache hit ratio is cached input / total input.

Scoring definitions are implemented and tested in `metrics.py`:

- CUAD-style normalized word-token precision/recall/F1 across all gold spans;
  this is **not the official CUAD AUPR leaderboard evaluator**.
- Clause/span precision/recall/F1 with one-to-one overlap matching; partial
  overlapping quotes cannot count as multiple recovered clauses.
- Exact answer match; absent accuracy; answerable F1. Missing/duplicate answers
  cannot earn a correct-absence point.
- Citation exact-location rate in the untouched source and character-overlap
  precision/recall/F1 against annotated intervals. If any gold span is nonliteral,
  marked `<omitted>`, or out of bounds, its question's character metrics are unknown
  and excluded from character averages; counts remain visible. Exact citation
  location is evaluated independently and does not prove relevance.
- Contract-cluster bootstrap 95% confidence intervals and paired comparisons
  aligned on `(contract_id, N)`, preserving repeated N rows from one contract in
  the same resampled cluster. Unmatched/duplicate pairs are rejected rather than
  silently discarded. Question position is the **one-based requested question order**,
  not the clause's location in the document; missing answers retain their position.

```sh
python -m benchmarks summarize benchmarks/results/run/runs.jsonl
```

This writes Markdown/CSV tables, a self-contained SVG, and question-position CSV.
Use `reporting.paired_comparison(rows, method_a, method_b)` for paired deltas and CIs.
Reports derived from actual results do not fabricate model
measurements from the estimator. Inspect locally before publishing: public source
contracts can still contain personal names/contact information. Default generated
artifacts are gitignored. Never publish credentials, private data, or raw SDK
exceptions. The CLI prints only allowlisted error codes/types.

## Prices and estimation

`prices.json` is editable; rates are USD per million tokens, separated into
`input_miss`, `input_hit`, `cache_write`, and `output`. Price sources checked on
**2026-10-24**:

| Example profile | Uncached input | Cache read | 5-minute cache write | Output |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek Flash, peak | $0.30 | $0.006 | $0.30 (no premium) | $1.20 |
| Claude Sonnet 4.6 | $3.00 | $0.30 | $3.75 | $15.00 |

Sources: [DeepSeek official pricing](https://api-docs.deepseek.com/quick_start/pricing),
[Anthropic official pricing](https://platform.claude.com/docs/en/about-claude/pricing).
DeepSeek off-peak rates are half the peak rates; the example deliberately uses
peak pricing. Sonnet 4.6 is an explicit price-tier example, not a promise about
future model availability. No batch discounts, taxes, or private pricing.

The tokenizer is a reproducible **proxy**, not DeepSeek/Anthropic's exact tokenizer.
Chat/tool serialization, reasoning tokens, provider minimum cache length, cache
write TTL, output verbosity, and repair/tool frequency can change the bill.
Full-document fork input is counted N times plus reduce; fork answers and reduced
answers both incur output. Tool estimation includes repeated accumulated history,
search/read payloads, and tool-call output. See generated estimate assumptions for
numeric output and repair assumptions and cold/warm/no-cache sensitivity.
Defaults assume 256 output tokens per answer and two tool loops (search + read).
Representative tool payloads are tokenized from actual selected contracts, not gold.
Each answer-bearing response has a 10% repair probability with up to two rounds:
expected additional calls per response = 0.1 + 0.01. Fork leaves and reduce,
every tool-agent question, and one-shot are all charged this contingency.
Compounded repaired history in later tool turns and additional repair tool loops
are excluded; review a larger contingency before authorizing an expensive run.
**An estimate is not a hard spending ceiling.**

### Offline sample estimate (2026-10-24)

Fixed sample document lengths: min 428, p25 2,694, median 3,873, p75 15,681.25,
p95 37,198.85, max 57,507, mean 10,914.20 `cl100k_base` tokens.
Full grid: 180 logical runs, about 214.824 million input tokens and 1.754 million
output tokens including the modeled repairs. No inference was performed.

| Method (20 contracts × all three N values) | DeepSeek cache usable | DeepSeek no cache | Sonnet cache usable | Sonnet no cache |
| --- | ---: | ---: | ---: | ---: |
| forkreduce | $1.4329 | $6.8204 | $21.6416 | $70.4546 |
| tool_agent | $3.0750 | $58.9941 | $81.2328 | $591.8269 |
| one_shot | $0.7120 | $0.7374 | $8.8601 | $8.4991 |
| **Full grid** | **$5.2199** | **$66.5519** | **$111.7345** | **$670.7805** |

Usable cache starts empty and includes population charges. Sonnet one-shot can
cost more with cache writes than without because there is little reuse. The
tool-agent cache case assumes ideal eligible-history reuse; the no-cache case
exposes the repeated-history cost. Neither establishes an accuracy ranking.
The generated estimate artifact contains the full per-N and per-contract grid.


## Five-model minimal live smoke

This command requires **separate explicit user approval**; it is not part of CI or
the offline PoC. It schedules availability probes and two short public CUAD
contracts, N=5, three methods, five configured model IDs. Unsafe models are refused
before sending; it never runs the full estimate grid.

```sh
FORKREDUCE_BENCH_ALLOW_PAID=1 python -m benchmarks.live_smoke availability --approve-paid-smoke
FORKREDUCE_BENCH_ALLOW_PAID=1 python -m benchmarks.live_smoke cuad --approve-paid-smoke
```

Set `OPENAI_BASE_URL` to the explicitly approved endpoint before running; the
repository contains no operator-specific relay URL. Runtime credentials come
from `PROXY_API_KEY`, then `OPENAI_API_KEY`, then `PROXY_API_KEY_FILE` (default
`~/.config/proxy_key`); neither the key nor SDK exception bodies are logged.
Results and the shared `budget.json` ledger live in ignored
`benchmarks/results/models-smoke/`.
Keep both phases in the **same directory**. Never run simultaneous processes
against one ledger.
Restart skips successful availability cases and completed CUAD cases, retaining
failed probe attempts. It never resets spending.
`--models` selects a subset without changing the shared $3 ceiling.

`provider_model(..., api="responses", settings={"prompt_cache_key": "...",
"extra_headers": {...}})` selects `OpenAIResponsesModel`; the default remains
chat completions. Pydantic AI's existing model settings reach fork/reduce without
core changes. SDK retries are disabled and requests use a custom User-Agent.
GPT-6.1-Sol and GPT-6-Luna use Responses and an explicit cache key; unavailable
`gpt-6.1-luna` is explicitly replaced by `gpt-6-luna`. The recorded Grok probes
compared no header with stable `x-grok-conv-id`; future Grok calls are now refused
by the budget guard, for the reason below. DeepSeek and MiMo use chat.
Both APIs' cached-token counters are normalized before validation and repair.
DeepSeek thinking rejects forced tool choice. Its benchmark capability profile
uses automatic tool choice and preserves `reasoning_content` across tool turns;
the same Answers schema, question-ID and citation validators still apply.
Setting `tool_choice="auto"` alone is insufficient because Pydantic AI resolves
tool-only outputs to required unless the profile declares forcing unsupported.
Its Chat profile sends the documented `max_tokens` field, not the OpenAI-specific
`max_completion_tokens` alias.

The request wrapper reserves a conservative no-cache input byte/framing bound
plus a **total billable output bound before every request**, including warmup,
repair, tools, reduce, and failed structured responses. OpenAI Responses/MiMo
have reasoning-inclusive caps; DeepSeek Chat conservatively reserves its official
384K absolute output ceiling while requesting only 2,048 tokens. Concurrent
reservations share one ledger. Missing input/output usage or transport uncertainty
retains the full reservation and stops sends. Measured usage and conservative
charged amounts remain distinct; interrupted reservations also stop sends.
After correcting a failed request, explicit `--resume-after-correction REASON_CODE`
can resume only fully charged uncertain reservations with no in-flight calls.
It keeps every charge and unknown-cost flag, adds an audit event, and cannot
resume a token-bound violation, underestimated reservation, or exhausted limit.

**Grok is not safe under a token-only hard spending gate.** Its official
[Chat API](https://docs.x.ai/developers/rest-api-reference/inference/chat-completions)
and [Responses API](https://docs.x.ai/developers/rest-api-reference/inference/responses)
explicitly exclude reasoning and function-call tokens from the visible-text cap.
The recorded smoke caught this mismatch and stopped; lowering reasoning effort
does not establish a hard bound. New Grok sends are refused even with a fresh
ledger. A verified provider/proxy monetary cap is a prerequisite for further runs.
`--quarantine-bound-model MODEL` can permanently isolate an observed output-only
violator while allowing independently bounded models to continue. It requires
known costs within the original reservations and no pending calls, preserves all
charges/audit records, and never relaxes input or monetary-bound violations.
This is a hard **local estimated-cost admission ceiling**, conditional on provider
honoring output caps, text tokenization/framing bounds, and the published rates;
it cannot enforce the proxy's undisclosed invoice or replace an account-side cap.
GPT unknown cache-population splits are conservatively priced at 1.25x normal
input, not silently treated as free writes. The pinned prices/sources are in
`live_smoke.MODEL_SPECS`; no full experiment is authorized by this command.

All methods request temperature 0 (reasoning providers may ignore it; SDK warnings
are not suppressed), identical schema and retry/repair limits, fork concurrency 1,
and output cap 2,048 for this smoke. Short-document selection is the two
shortest contracts with 2,000–7,999 proxy tokens in the prior seeded 20-contract
sample; it does not inspect gold labels. This is pipeline/cache evidence, not a
representative accuracy estimate. Per-model reports avoid pooling model scores;
combined `summary.csv` retains a row per model × method.
JSONL retains every attempt; summaries select the latest attempt per model,
contract and method. Non-executed safety-blocked cells have unknown accuracy,
not fabricated zero-cost model results or ordinary failed model generations.

## Paid execution: only after separate approval

**The original offline PoC did not authorize paid calls.** Before a future grid,
show the complete estimate and assumptions to the user and obtain explicit
approval. The general grid runner accepts provider secrets through environment variables:

| Provider | Required | Optional |
| --- | --- | --- |
| OpenAI-compatible | `OPENAI_API_KEY`, `OPENAI_MODEL` | `OPENAI_BASE_URL`; repository wrapper also recognizes `FORKREDUCE_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Provider caching settings are documented in `providers.py` |

Do not place keys in config files, CLI arguments, shell logs, or PRs. SDK hidden
retries are disabled. After approval, an operator must additionally set
`FORKREDUCE_BENCH_ALLOW_PAID=1` and pass the exact estimate artifact SHA256 plus a
positive USD stop threshold. The following is documentation, **not executed here**:

```sh
python -m benchmarks run --provider openai \
  --estimate-file benchmarks/results/estimate/estimate.json \
  --approve-estimate APPROVED_ESTIMATE_SHA256 \
  --price-profile deepseek-flash --budget-usd APPROVED_USD_THRESHOLD
```

The workload and prices must still match the approved artifact. Choose the price
profile corresponding to the actual model; examples are not interchangeable.
Set the intended provider/model environment first, regenerate the estimate, then
obtain approval: its workload snapshot also binds provider, model identity, and
the selected price profile. Changing any of these invalidates approval.
The threshold is checked **after each billed workload**, so one workload can
overshoot it. Unknown usage stops further work. Set a provider/account-side hard
limit separately if required. Existing `runs.jsonl` is never silently overwritten.
