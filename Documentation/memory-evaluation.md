# Memory retrieval evaluation

This is a reproducible **development benchmark of context retrieval**, using 40 hand-labeled queries over 33 synthetic coding-project records. It measures whether the memory component supplies useful, permitted records. It does not measure whether an LLM follows them or finishes a coding task.

The saved run is [memory-evaluation-results.json](memory-evaluation-results.json). Query text, scope, expected keys, returned keys, latency and token estimates are included for every method. The fixture is [memory_retrieval.json](../tests/fixtures/memory_retrieval.json); the runner is [evaluate_memory.py](../scripts/evaluate_memory.py).

## Reproduce the offline run

From the repository root:

```sh
python -m scripts.evaluate_memory --output Documentation/memory-evaluation-results.json
```

Use `python3` instead of `python` where required. The default run needs only the project's Python environment and SQLite with FTS5. It makes no network or LLM calls, reads no API credentials, and never opens the user's memory database. It creates and removes a temporary database, seeds all records, applies actual update/supersede/delete operations, then evaluates the queries.

Default settings are top-k = 3, estimated context budget = 1,200 tokens, one untimed warm-up and three timed repetitions per query. Settings can be changed explicitly:

```sh
python -m scripts.evaluate_memory --top-k 5 --max-tokens 1200 --repeats 5
```

The output records the configuration, Python/SQLite versions and a SHA-256 of the canonical fixture JSON. Timestamps and latency vary between runs. Compare relevance metrics using the same fixture, settings and implementation version; run after code changes to obtain current evidence.

## Dataset and labeling

The fixture has 12 simple, 18 moderate and 10 difficult queries. There are 29 answerable queries and 11 queries whose correct relevant-memory set is empty.

| Area | Examples |
| --- | --- |
| Exact and identifier search | CPython version, FastAPI path, OAuth/PKCE, FTS5 |
| Execution evidence | ModuleNotFoundError, EADDRINUSE, datetime TypeError |
| Natural language and multiple relevant records | Regression command, dependency installation, atomic corrections |
| Scope | Project, user, omitted user, agent, session, task and global visibility |
| Lifecycle | Expired, disputed, deleted, superseded and unverified records |
| Difficult meaning-based queries | Identity verification, persistence across launches, timestamp comparison |
| Abstention | Unrelated aquarium question; queries for inaccessible or inactive records |

Expected and forbidden records use stable fixture keys, not database IDs. The runner also independently treats every inactive or inaccessible record as forbidden. Scope/lifecycle checks therefore remain visible even when a wrong answer happens to have lexical overlap.

Labels were authored while developing the component. They are not independently adjudicated, sampled from production, or hidden from development. Most facts have a single relevant answer, one query has two, and irrelevant recent records are deliberately present. These choices make this a diagnostic fixture; its distribution is not a claim about typical user traffic.

## Compared methods

1. **No memory:** selects no records. This establishes retrieval coverage without persistent context. It cannot estimate how much an LLM knows from its parameters or current task.
2. **Recent-record proxy:** selects the newest active, permitted records, independent of query, within the same top-k and budget. This is a controlled analogue of sending recent raw context. It is **not an actual raw-conversation-history baseline**: it contains curated memory records, omits role/tool structure and summaries, and inherits scope/lifecycle safeguards. The three recent task outcomes intentionally displace older useful requirements.
3. **Scoped FTS5:** calls the real `MemoryStore.retrieve`, including its metadata filters, status/expiry policy, FTS5 ranking and metadata signals. All output is subsequently rendered into the same source-bearing format and bounded by the same budget as the recent-record proxy.

The context representation is JSON containing record ID, type, source reference and content. Budgeting includes this representation, not just the fact text. Estimated tokens use `ceil(UTF-8 bytes / 3)`, matching the application's heuristic. This is neither a model tokenizer nor a billed token count. It measures retrieved context only; the system prompt, current user request and full conversation are excluded.

## Metrics

For answerable queries, let `R` be the returned records and `G` the hand-labeled relevant set:

- **Precision over returned records:** `|R ∩ G| / |R|`, or zero when no record is returned.
- **Fixed-k precision@k:** `|R ∩ G| / k`. This conventional fixed denominator is reported separately so returning one correct record is not mislabeled as precision@3 = 1.
- **Recall@k:** `|R ∩ G| / |G|`.
- **MRR@k:** reciprocal of the first relevant record's rank, or zero if absent.

Each metric is macro-averaged over the 29 answerable queries. Empty-label queries are excluded from those averages and scored by **no-answer accuracy**: the fraction that return no records. **Exact-set accuracy** includes all 40 queries and requires the returned set to equal the labeled set. **Forbidden-hit counts** expose any returned scoped-out or inactive records independently of relevance.

Latency is the median of three warm runs per query; the report gives the mean and nearest-rank p95 of those query medians. It includes retrieval, rendering and budget selection. It excludes seeding, cold process/database startup and LLM generation. The recent-record proxy uses an already loaded list, so it is a context-selection comparison, not a storage-engine latency comparison.

## Recorded results

The saved offline run was executed on 27 September 2026 UTC with Python 3.14.4 and SQLite 3.53.0 on Darwin arm64. No API was used and no semantic quality result is claimed.

| Method | Precision over returned | Fixed P@3 | Recall@3 | MRR@3 | No-answer accuracy | Mean retrieved token estimate | Mean warm latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No memory | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0 | 0.0001 ms |
| Recent-record proxy | 0.0115 | 0.0115 | 0.0345 | 0.0172 | 0.0000 | 156.0 | 0.0223 ms |
| Scoped FTS5 | 0.8046 | 0.3103 | 0.8966 | 0.8966 | 0.9091 | 52.5 | 1.5678 ms |

All three methods returned **zero forbidden records**. FTS5's exact-set accuracy was 0.7750 and p95 warm latency was 2.6647 ms. Its maximum selected-context estimate was 200, below the 1,200 budget. The no-memory method's perfect abstention follows directly from selecting nothing and should not be confused with task success.

FTS5 retrieves the labeled answer for all simple and moderate answerable queries. It only retrieves one of four difficult answerable queries, giving difficult-case recall = 0.25. The misses are retained in the fixture:

- `q34`, “How do people prove who they are?” misses the OAuth/PKCE fact.
- `q35`, “Recall our choice for remembering things between launches” misses SQLite persistence.
- `q37`, “How do we keep wall clock representations comparable?” misses the UTC datetime fix.

Several queries return the correct record followed by irrelevant lexical matches. For example, `q03` also selects a lint command containing `memory`. Query `q40` excludes the global credential rule as requested but selects the unrelated terminal-output record because it mentions logs. This reduces no-answer accuracy without constituting a scope leak.

The results support selective retrieval over a small recent-context proxy on this fixture. They also expose the limit of keyword matching on paraphrases. They do not prove semantic search would solve these cases, establish general retrieval quality, or establish improved coding-task success.

## Optional real hybrid experiment

The same runner can compare the optional OpenRouter embeddings adapter with FTS5. This path is explicitly networked, sends the synthetic query/eligible fixture content to the provider and may incur provider charges. Set `OPENROUTER_API_KEY` in the environment and choose an available model; the runner does not load or print credentials.

```sh
python -m scripts.evaluate_memory \
  --embedding-model "$MEMORY_EMBEDDING_MODEL" \
  --repeats 1 \
  --output Documentation/memory-evaluation-hybrid-results.json
```

Set `MEMORY_EMBEDDING_MODEL` to an explicit provider model identifier before running. The checked-in results were generated **without** this flag. No synthetic vector provider is used to claim semantic accuracy.

The optional report adds `scoped_hybrid`, the provider fingerprint, request/text counts, exception-type counts and whether fallback occurred. Provider errors leave keyword retrieval and any usable cached matches available; degraded results must not be described as evidence of successful semantic retrieval. An untimed warm-up generally absorbs both document embeddings and the query embedding, which now has a process-local LRU cache. Each query reports `timed_embedding_requests`; timed cache misses or retries include provider latency. Total provider counts include the warm-ups. This does not measure cold embedding cost, total provider billing or token usage. A local provider implementing `EmbeddingProvider` can also be supplied to the runner's `evaluate()` function for an offline semantic experiment.

## Audit regression coverage

The September memory fixes retain the relevance scores above. Their effects are exercised separately in `test_memory_regressions`, `test_memory_loop_regressions`, `test_memory_privacy`, `test_tool_observations`, and `test_memory`:

- A resumed unfinished task retains identity, plan, artifacts, retries and task-scoped records through the actual `run_turn` loop.
- Twelve 3,500-character file results compact within the default history budget, preserving tool protocol and the newest complete evidence. Model/tool responses are mocked.
- A real sequence of failure, read, write, restart and successful rerun creates a qualified recovery lesson linked to the error and changed file.
- Large plans and a large standing requirement coexist within the exact injected-envelope budget. A four-fact probe (`Use Python 3.12`, `Use SQLite`, `Run pytest`, `Keep API stable`) uses 175 estimated tokens including task state and provenance; the fact text alone is 18 estimated tokens.
- Late error identifiers, cached semantic records beyond 256, query reuse and partial embedding failures have deterministic regression tests. Synthetic vector tests measure mechanics, not semantic quality.
- Source expressions survive the next model request, real credential literals are redacted, source pages reconstruct long files, and memory search retains complete JSON above 4,000 characters.

These are targeted component/integration guarantees, not evidence that a live LLM will choose the correct memory action or complete an unseen task.

The second audit adds source/config and paging-boundary privacy checks, quote-level correction, standing-fact/lesson allocation, evidence reconfirmation, uniform UTF-8 budgeting and trusted runtime restore notices. The complete suite now has 161 passing tests. See [the requirements audit](memory-requirements-audit.md) for reproducers and limits. The saved retrieval run above is still entirely offline; a separate [live memory-only attempt](memory-live-smoke-results.json) could not validate model behavior because the service returned HTTP 429 during connectivity diagnosis.

## Relationship to the assignment

Lecture 2 PDF p9–14 (slides 18–27) motivates selective storage, indexing, retrieval, updating/forgetting, reliability, scope and coding-agent memories. This benchmark provides component evidence for those choices.

The assignment's p4 requirement for **at least 30 test tasks** still needs a separate end-to-end coding evaluation. These 40 retrieval queries must not be counted as 40 completed coding tasks. The final system evaluation should use frozen development/held-out splits, simple/moderate/difficult coding goals and failure-inducing cases; compare single-agent and multi-agent or another simpler baseline; record success, output quality, tool success, iterations, latency, token usage/cost, failure and recovery. A separate memory ablation should compare the same agent/model/tools with no persistent memory, actual conversation history and the selected memory policy.

Cross-session agent adherence, prevention of repeated failed actions, LLM prompt-injection resistance, long conversations, concurrent load, large-corpus retrieval and durable sandbox recovery are not established by this fixture. Use the component tests and integration tests for their specific guarantees, and do not substitute retrieval scores for those broader claims.
