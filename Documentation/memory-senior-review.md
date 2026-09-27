# Practical coding-agent memory review

Initial review: 27 September 2026. Remediation verified: 28 September 2026 (Australia/Melbourne). The original findings below describe the **pre-fix implementation**; old line numbers are historical. The focus is practical usefulness, correctness and efficiency for coding tasks, rather than matching the earlier design document.

**Second audit:** a fresh review against the assignment and lecture found further source/config filtering, page-boundary, correction, standing-fact allocation, evidence-refresh, Unicode-budget and runtime-environment issues. These have been fixed with additional regression tests. The current suite is **161 passing tests**. Read the [current requirements audit and fixes](memory-requirements-audit.md) for the latest conclusions; the 132-test remediation snapshot below records the first repair round. A live model attempt was unavailable (diagnostic HTTP 429); the current machine also lacks `minikube`. Neither is presented as successful live validation.

## Remediation status

The confirmed failures below now have implementation fixes and offline regression coverage. SQLite remains the default. No hosted database, new infrastructure or paid API is required.

| Finding | Implemented behavior | Regression evidence |
| --- | --- | --- |
| 1. Source corruption | Shared credential filtering preserves annotations, input/environment expressions and prompt string boundaries; real credential literals remain redacted | `test_memory_privacy`; next-model-request test in `test_memory_loop_regressions` |
| 2. Resume wipes state | `begin_turn` continues unfinished tasks; `/task` explicitly starts new work, `/continue` also continues answered work | Resumed identity/plan/artifacts/retries/task-scope and explicit-new-task loop tests; conflict rollback tests |
| 3. Empty recall after budget packing | Fit working state first, reserve recall space, then trim more state if the first complete fact needs it | Large plan and large requirement tests in `test_memory_regressions` |
| 4. Late error identifiers lost | Select informative terms throughout the query; automatic recall includes the active step and up to two unresolved failures through subsequent reads/edits | Late error/path tests in `test_memory`; unresolved-error recall test |
| 5. Long task cannot compact | Evict completed tool exchanges inside the current turn; preserve the current user anchor, latest/pending exchange and bounded observational summaries | Twelve large read results through `run_turn`; parallel pending-group and explicit overflow tests |
| 6. Edit/retest misses recovery | Persist bounded pending failures per task/command; later success links previous error and intervening written paths without claiming causality | Failure → read → write → restart → success regression; new-task isolation and failed-write tests |
| 7. No grounded correction path | `memory_correct` validates an exact current-user quote, enforces ownership and atomically supersedes a predecessor. Obvious correction wording bypasses append-only auto-write | Correction, invented quote, foreign scope, boolean ID and latest-user grounding tests |
| 8. 256-record semantic coverage cliff | Stream all eligible cached vectors; limit only new embedding ingestion and retained candidates/cache size | Older semantic match beyond 256 peers in `test_memory` |
| 9. Wasted embedding work | Validate/cache query first; persist document batches independently; retain cached matches on later provider failure | Query failure, partial batch, repeat-query and provider-change tests |
| Structured output and file clipping | Memory results use their own exact envelope budget; generic observations keep valid JSON and status; `read_file` exposes resumable line/column pages | >4,000-character memory result in the loop; reconstruction and status tests in `test_tool_observations` |
| Prompt overhead | Prompt retains record ID/type/content/source kind/reliability; full provenance and ranking stay in `/trace` | Four-fact context and diagnostic separation test |

Verification after fixes:

```sh
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m scripts.evaluate_memory --output Documentation/memory-evaluation-results.json
git diff --check
```

- **132 tests passed**, compared with 81 at the initial review. The new tests include runtime-loop regressions with mocked model/tool responses.
- The 40-query development benchmark retains recall **0.896552**, returned-record precision **0.804598**, and **zero forbidden hits**. Saved warm FTS retrieval averages **1.1100 ms** on this small fixture. See [the evaluation and its limitations](memory-evaluation.md).
- A new four-short-fact prompt probe uses **175 estimated tokens**, including working state and essential record provenance. It is a deterministic byte-based estimate, not a provider token measurement or a like-for-like comparison with the older overhead probe below.
- A synthetic warm hybrid probe with 256 × 1,536-dimensional vectors measured **22.39 ms median** over ten repeated queries, with **zero additional provider calls**. This measures local cache/search mechanics, not semantic accuracy or real network latency.

Remaining limits are explicit: source filtering is best-effort; language understanding and choosing the right correction still depend on the model; one oversized latest/pending exchange fails explicitly instead of silently breaking protocol. Semantic comparison scans eligible cached vectors and is linear in corpus size × dimensions, so larger deployments need measured capacity planning. Repository-derived facts are still not automatically promoted from entire source files: the durable automatic policy records command/error/recovery evidence, while arbitrary facts require user-grounded writes. Live LLM adherence, sandbox execution/restoration and the assignment's 30 end-to-end coding tasks remain separate validation work.

## Initial verdict (before fixes)

The persistence foundation is useful: local SQLite/FTS5, scoped access, atomic corrections, explicit provenance and offline tests. The application layer is not yet robust enough for sustained coding tasks. Passing the current tests does not establish reliable resume, long-task compaction or recovery learning: additional isolated probes found failures in those paths.

The highest-value next step is to fix context/state behavior and improve realistic tests. Switching to Supabase would not address the confirmed problems.

## Confirmed correctness issues

### 1. P1: credential filtering corrupts ordinary source code

Location: [memory/memory.py](../memory/memory.py), `_CREDENTIAL_ASSIGNMENT` around lines 49–53; [main.py](../main.py), line 121; [memory/service.py](../memory/service.py), `clean_context` and `checkpoint`.

This is valid source:

```python
def validate(password: str) -> bool:
    password = input("Enter password: ")
    return bool(password)
```

The redaction pattern treats the type annotation and expression as credential values. The resulting observation includes `password: [REDACTED] -> bool:` and `password = [REDACTED]"Enter password: ")`, which no longer compile. A mocked `run_turn` confirmed that the next model request receives the damaged file content. This can make the agent misunderstand or reproduce broken authentication code.

Narrow detection to actual credential values rather than arbitrary identifiers/expressions, and distinguish safe durable serialization from faithful active code observations. Real secret values still need explicit protection; disabling all filtering is not the fix.

### 2. P1: continuing a resumed session resets its restored task

Location: [main.py](../main.py), `run_turn` line 96; [memory/service.py](../memory/service.py), `start_task` lines 204–209.

Reproducer: save an unfinished pagination task with a plan, current step, artifact and retry count; load the session in a new service; submit `Continue`. Before the first model call, the restored task is replaced by a new ID with goal `Continue`, empty plan/artifacts, no active step and zero retries. The old conversation survives, but the restored blackboard is discarded and records scoped to the previous task ID become invisible.

Separate continuing an unfinished task from creating a new task. Preserve task identity and working state on resume, with explicit semantics for starting new work.

### 3. P2: budget packing can discard every memory and then leave the budget unused

Location: [memory/service.py](../memory/service.py), `build_context` lines 383–400.

The service tries to add retrieved records while the base task/summary is oversized. It subsequently removes the task/summary but never tries those records again. A valid plan of 12 items × 480 characters plus a 480-character active step changes retrieval of a saved pytest requirement from record #1 to `task: null, memories: [], summary: []`. The returned envelope uses only **50 of the default 1,800 estimated tokens**.

Bound/compact the base working state first, reserve space for critical requirements, then select retrieved records. Repack after removing lower-priority material.

### 4. P2: useful identifiers at the end of a request can be dropped

Location: [memory/memory.py](../memory/memory.py), `_fts_query` lines 833–838; [main.py](../main.py), lines 103–105.

Only the first 30 unique non-stopword tokens become the FTS query. A roughly 345-character coding request ending `First resolve EADDRINUSE.` failed to retrieve the saved EADDRINUSE fix, while the focused query retrieved it. Automatic recall also repeats the original user goal on every model round, rather than incorporating the current plan step or new tool error.

Select informative errors, identifiers and paths across the entire request. Build recall queries from the goal plus the current step/latest failure. Preserve the agent's explicit search tool as an additional path.

## Important operational gaps

### 5. P2: a single coding task cannot compact its completed tool exchanges

Location: [memory/service.py](../memory/service.py), `_compact` lines 171–187.

Groups are split only at user messages, so an autonomous multi-step task remains one indivisible group. With default history budget 6,000, a single goal followed by 3,500-character file-read results fails on the fifth result with `ContextBudgetExceeded`; its summary stays empty. This behavior is documented, but it materially limits normal coding work before the 12-round execution cap.

Compact completed tool exchanges within the active turn, retaining the goal, plan, current evidence and references to artifacts. Preserve any pending call/result protocol group. Increasing the budget alone delays the problem and increases repeated prompt cost.

### 6. P2: the ordinary edit-and-retest workflow creates no recovery lesson

Location: [memory/service.py](../memory/service.py), `observe_tool` lines 318–346.

`pytest fails → read_file → write_file → pytest succeeds` produces an error and an episode, but no lesson. Each read/write replaces the single `latest_observation`, so recovery detection only works for immediately consecutive retries of the same command. The current mocked recovery test omits the actual editing steps.

Track unresolved failures by task and normalized command across intervening tools. Link later success and relevant artifacts, while retaining the distinction between observed recovery and proven cause.

### 7. Natural-language corrections can leave contradictory active requirements

Location: [memory/service.py](../memory/service.py), `observe_user` and `tool_definitions`.

Remembering SQLite exclusively, then remembering PostgreSQL exclusively instead, leaves both facts active at reliability 1.0 and eligible for standing context. Manual `/correct` is atomic and works; the normal conversation flow has no grounded correction action. This is a missing interaction capability, not a failure of the low-level supersession transaction.

Support evidence-grounded replacement with a predecessor ID and distinguish corrections from independent facts. The agent also currently lacks a durable fact-writing path for verified repository knowledge discovered through files/tools: long-term fact writes require current-user quotes, while successful file reads remain session context.

## Optional hybrid-search findings

These issues affect opt-in embeddings, not the default offline FTS path. All probes used deterministic synthetic providers, not real APIs.

### 8. P2: a query-independent 256-record cap hides older semantic matches

Location: [memory/memory.py](../memory/memory.py), `_semantic_candidates` lines 569–574.

A saved SQLite-persistence fact matched the paraphrase `retention between launches`. Adding 256 unrelated records with the same importance made the same query return no result, because the target no longer entered the candidate window even though its embedding was already cached.

Separate the budget for embedding new records from the coverage of similarity search. Search eligible cached vectors by similarity; avoid using recency/importance as the sole gate before semantic comparison.

### 9. P2: a query-embedding failure discards successful document embedding work

Location: [memory/memory.py](../memory/memory.py), lines 592–614.

The code embeds missing documents, then the query, and only then persists document vectors. With 64 records and three synthetic query timeouts, calls `[32, 32, 1]` repeat three times: **192 document embeddings are recomputed and the cache remains empty**. Keyword fallback works, but this retry pattern wastes provider work when enabled.

Validate the query first, persist valid document batches independently with model/dimension checks, and reuse identical query vectors within a task where appropriate. Do not hold a database transaction during remote calls.

## Efficiency measurements and secondary improvements

These are local diagnostic measurements, not production SLOs or billed model-token counts.

| Probe | Result |
| --- | --- |
| Default FTS5, 1,000 synthetic command episodes | ~3.26 ms median warm retrieval |
| Default FTS5, 10,000 episodes | ~10.58 ms |
| Default FTS5, 50,000 episodes | ~44.88 ms |
| Cached hybrid, 256 records × 1,536 dimensions | ~203 ms local work, excluding real network latency |
| Four short standing requirements | 28 estimated tokens of fact text, 473 in the complete injected envelope with task/provenance/metadata |
| Five identical cache-warm semantic queries | Five additional query-embedding provider requests |

SQLite is adequate for the expected local workload. A leaner model-facing representation could keep IDs, concise content and essential provenance while leaving verbose ranking explanations and internal state fields in `/trace`.

`memory_search` currently passes through the same generic 4,000-character truncation as terminal output. A valid 4,487-character search response containing six records became an `excerpt` string with a truncated JSON tail. Structured memory search should select complete records within its own budget instead of clipping serialized JSON. Similarly, a file observation of 4,050 characters is reduced to 1,500, whereas 3,950 characters are preserved. Preserve useful diagnostics and expose a way to retrieve omitted portions rather than applying a coarse size cliff.

## What was verified

- Existing automated suite rerun: **81 tests passed**.
- Existing 40-query development retrieval benchmark rerun: recall 0.896552, returned-record precision 0.804598, fixed precision@3 0.310345, zero forbidden hits.
- Additional temporary-database and mocked-run probes reproduced the issues above without modifying user memory or invoking an LLM, remote embedding service or sandbox.
- The measured local corpus/embedding benchmarks were bounded and synthetic; network latency and live agent task success were not measured.

The existing benchmark is useful component evidence, but 33 curated records and short queries do not test active-task compaction, learned-memory quality, the 256-record cap or real repair workflows. Its bug-fix facts are prewritten, not generated by the current observation policy.

## Recommended order

1. Correct source redaction, preserve resumed task state, and repair budget packing.
2. Support compaction within a task and track failures across actual edits.
3. Improve task/error-driven retrieval and grounded memory corrections; keep model context concise.
4. Fix hybrid candidate coverage and caching before enabling it for normal usage.
5. Evaluate repeated coding tasks across sessions with memory on/off, measuring task success, repeated mistakes, actual input tokens and elapsed time.

A stronger coding-agent memory needs fewer stale or irrelevant facts and better continuity, rather than simply a larger database or more memory categories.
