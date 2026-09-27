# COS30018 — Coding Agent Harness

A Python coding-agent CLI with persistent, scoped memory. It remembers explicit requirements across sessions, restores conversation/task checkpoints, retrieves prior tool observations and keeps model context bounded. SQLite and FTS5 run locally; a hosted database is not required.

The interactive runner currently uses the **coder**. A reviewer adapter shares its memory/tool protocol, but planner/executor roles remain stubs and a complete multi-agent coordinator is future work. See [implementation and limitations](Documentation/memory-implementation.md) and [retrieval evaluation](Documentation/memory-evaluation.md).

## Setup

Use Python **3.10+**, with SQLite compiled with FTS5:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python main.py
```

On Windows, activate with `.venv\Scripts\activate`. Run from the project directory: the default project identity includes its resolved path. The default database is `memory/memories.db`, ignored by Git. `.env` is also ignored; keep its API key private.

Memory commands work with **no API key and no Minikube**. To request model responses, set `OPENROUTER_API_KEY` in `.env` and optionally choose a tool-capable `OPENROUTER_MODEL`.

Filesystem and terminal tool calls additionally require `minikube`, `kubectl` and a working Minikube driver. The application starts the configured profile and creates a temporary sandbox pod only when a filesystem/terminal tool is first called. The default image is `python:3.12-slim`; use `SANDBOX_IMAGE` for a prepared image with the dependencies your coding tasks need. See the [Minikube installation guide](https://minikube.sigs.k8s.io/docs/start/).

Project files are copied into `/home/user/project` in the sandbox. Environment files, private-key file extensions, database files, the configured memory database, Git metadata and several local dependency/configuration directories are excluded. These exclusions are a best-effort boundary, not a general secret scanner. Sandbox edits are temporary; the current runner does not automatically export them back to the host.

## Try persistent memory without an API

Start `python main.py`, then enter:

```text
/remember:requirement Use Python 3.12 for this project.
/remember:constraint HTTP requests must use a 60-second timeout.
/memories
/search Python runtime
/state
/sessions
exit
```

Restart and enter `/search Python runtime`: the saved fact persists. Each fresh launch creates a new conversation unless `--resume SESSION_ID` is supplied; durable facts remain available in the same project/user scope.

```sh
python main.py --resume SESSION_ID
```

For a model-assisted demonstration after configuring the API, send `Remember that this project must use Python 3.12.`, then `/new`, then ask it to implement an endpoint. Explicit durable intent can be captured automatically. Standing requirements/preferences/constraints are considered even without query keyword overlap, using a budget-derived candidate limit capped at 64. Packing reserves space for query-relevant recall before filling unused space with more standing facts; inclusion remains subject to the configured memory budget.

Repeating the same fact from an equally or more reliable, authoritative source renews its recency, provenance and eligible lifetime without adding a duplicate. Weaker sources cannot override that evidence. For a changed fact, the agent can use `memory_correct` with a quote from your latest message. Correction checks inspect the quoted fact and possible local predecessors, so an unrelated request containing “replace” does not block saving an independent requirement.

## CLI commands

| Command | Purpose |
| --- | --- |
| `/remember <text>` | Save a decision with user provenance |
| `/remember:<type> <text>` | Save a typed requirement, preference, constraint or decision |
| `/memories` | Inspect visible durable records and statuses |
| `/search <query>` | Inspect the bounded recall envelope; `/trace` shows full provenance and ranking reasons |
| `/correct <id> <text>` | Atomically supersede an owned fact and retain its prior version |
| `/forget <id>` | Delete an owned durable fact |
| `/sessions` | List recent sessions in the current project/user scope |
| `/new` | Create a fresh conversation while retaining durable facts |
| `/resume <id>` | Restore an existing conversation and task checkpoint |
| `/task <goal>` | Start a new task in the current conversation |
| `/continue [message]` | Continue the current task, including one that already returned an answer |
| `/delete-session <id>` | Delete its transcript, summary, checkpoint and events; use `/new` first for the current session |
| `/state` | Inspect the structured current task and saved plan |
| `/trace` | Inspect recent memory/model/tool events |
| `/help` | Show command help |
| `exit` or `quit` | Exit and terminate any sandbox pod created by this process |

Deleting a fact and deleting a session are separate operations. A forgotten fact may still appear in its original transcript; deleting a session does not delete reusable facts extracted from it. Deletion is logical application deletion, not guaranteed forensic erasure from storage or backups.

Resuming restores memory, **not sandbox files or processes**. Both `/resume` and `--resume` expose this limitation, and every restored-session model request receives an application-owned environment notice even when the historical-memory budget is too small to retain task details. Interrupted tool calls are marked with an unknown outcome and are never automatically replayed. Inspect the live workspace before retrying an action. Multiple workers cannot silently overwrite the same session checkpoint: a stale revision requires reloading with `/resume`.

A normal message continues an unfinished task and retains its goal, plan, active step, artifacts, retry count and unresolved command failures. After a task is answered, a normal message starts a new task; use `/continue` to keep working on the answered task or `/task <goal>` to explicitly replace unfinished work. `/new` starts a separate conversation.

## Configuration

All values are optional except the API key when calling a model. See [.env.example](.env.example).

| Variable | Default / meaning |
| --- | --- |
| `OPENROUTER_API_KEY` | Blank; required for chat or opted-in remote embeddings |
| `OPENROUTER_MODEL` | `openrouter/free`; select a model supporting the requested tools |
| `MEMORY_DB_PATH` | `memory/memories.db` in this repository; `--db PATH` takes precedence |
| `MEMORY_PROJECT_ID` | Resolved current-directory name plus path hash |
| `MEMORY_USER_ID` | Current OS username |
| `MEMORY_HISTORY_TOKENS` | `6000`; estimated recent-history budget |
| `MEMORY_CONTEXT_TOKENS` | `1800`; estimated final historical-memory envelope budget |
| `MEMORY_EMBEDDING_MODEL` | Blank disables remote embeddings; explicit model ID opts in |
| `AGENT_MAX_ROUNDS` | `12`; maximum model/tool rounds per submitted request, including a continuation |
| `MINIKUBE_PROFILE` | `minikube` |
| `SANDBOX_NAMESPACE` | `agent-sandbox` |
| `SANDBOX_IMAGE` | `python:3.12-slim` |

Token budgets consistently use `ceil(UTF-8 bytes / 3)` in low-level retrieval, history and prompt packing, not a model-specific tokenizer. The history and memory budgets are separate and do not account for the entire model request. Older turns and completed tool exchanges within a long task become bounded extractive summaries. The current user message and latest or pending tool exchange remain intact. If those alone exceed the history budget, use smaller file pages/edits or increase `MEMORY_HISTORY_TOKENS`.

The `read_file` tool accepts optional `start_line`, `max_lines` and `start_column` arguments. It filters the complete file before pagination, up to a **2 MiB file limit**, so a page boundary cannot hide a credential's identifying prefix. Equal-length masking preserves line/column cursors. Results include `path`, `redacted`, and continuation coordinates `next_line`/`next_column`; `redacted=true` means the returned content differs from the file. Larger files require targeted command inspection. Filtering is repeated for each page: a local probe near the size cap took about 1.5 seconds per page, so this is a correctness trade-off rather than an optimized large-file reader.

Terminal output is filtered before head/tail clipping, retaining final diagnostics without exposing a secret because its label was clipped away. Working-state summaries are reduced before recall is packed so a large plan does not crowd out all relevant facts. Memory tools return their already-budgeted JSON directly; generic tool-output truncation does not cut `memory_search` results.

Setting `MEMORY_EMBEDDING_MODEL` enables the real OpenRouter adapter. Eligible memory text and queries then leave the machine and provider charges may apply. Keyword retrieval is the default and remains the fallback on embedding failures. Semantic retrieval scans all eligible cached vectors, bounds new embedding work separately, saves successful batches immediately and reuses a bounded in-process query cache. Remote embeddings have contract tests but were not live-tested for this implementation; the `EmbeddingProvider` interface also supports a local model. The memory database is not encrypted and project/user labels are not authentication for a hosted multi-tenant service.

Credential validation and redaction share `memory/privacy.py`. File filtering selects source or configuration behavior from the path, preserving tested source annotations, variable references, function calls and JSON schemas while redacting recognized credential literals and nested values. Configuration values are handled separately so a YAML password is not mistaken for a source annotation. For ambiguous untyped text, bare assignments such as `api_key=short` are treated as configuration secrets. Filtering remains best-effort and cannot identify arbitrary secrets reliably.

## Tests and evaluation

```sh
python -m unittest discover -s tests -v
python -m scripts.evaluate_memory --output Documentation/memory-evaluation-results.json
```

Automated tests use temporary databases and mocked model/sandbox/provider responses. They cover persistence, isolation, exact-quote corrections, repeated-fact renewal, standing-fact allocation, source fidelity, task continuation, concurrency, budgets, compaction within long turns, interrupted tool groups, failure/edit/recovery lessons, credential-safe pagination, embedding caches and agent integration. They do not require API calls or a running cluster. A separate two-process CLI smoke test verifies save/restart/search without a model or sandbox.

During the 28 September audit, a live request using the configured `openrouter/free` route returned HTTP 429. The local `minikube` executable was also unavailable, preventing a real coding-runner check. These attempts provide no evidence of live memory adherence or coding-task quality; the successful checks above establish component behavior only.

The offline benchmark contains 40 labeled **retrieval queries**, including paraphrases, stale records, scope exclusions and no-answer cases. It compares no memory, a bounded recent-record proxy and scoped FTS5. It is a development benchmark, not unseen coding-task evidence or fulfillment of the assignment's minimum 30 end-to-end test tasks. See the [recorded results and metric definitions](Documentation/memory-evaluation.md).

## Project map

- [Memory implementation](Documentation/memory-implementation.md): architecture, lecture mapping, decisions and limitations.
- [Assignment/lecture audit](Documentation/memory-requirements-audit.md): verified requirements mapping and the evidence still needed for the whole assignment.
- [Week 3 research](Documentation/week-3-memory-research.md): historical design record.
- `memory/memory.py`: SQLite records, FTS5 retrieval, scoping, lifecycle and optional hybrid ranking.
- `memory/service.py`: sessions, task state, compaction, write policy, recall and memory tools.
- `memory/context.py`: token estimates and compaction of complete tool exchanges.
- `memory/privacy.py`: shared source-aware credential filtering and validation.
- `memory/observations.py`: bounded structured tool-output excerpts.
- `memory/prompt.py`: shared historical-memory serializer.
- `memory/embeddings.py`: optional embedding protocol and OpenRouter adapter.
- `main.py`: interactive coder loop and memory commands.
- `agent_roles/`: coder/reviewer request adapters and unfinished planner/executor roles.
- `scripts/tools/`: filesystem/terminal tools and sandbox setup.
