# Memory v3

The goal is simple: keep the current conversation, retain selected facts across conversations, and let the coder call memory tools when useful. Storage and execution use LangGraph's native interfaces.

## Framework structure

```text
new user message -> coder -> tools -> coder -> final response
                     |        |
                     +--------+-- SqliteSaver: conversation checkpoints
                              +-- SqliteStore: durable memory tools
```

`agent_roles/graph.py` defines a two-node `StateGraph` compiled with both a checkpointer and a store. This replaces the manual model/tool loop because LangGraph must own state updates to checkpoint each graph step and resume unfinished work. The existing OpenRouter request function, tool dispatcher, and sandbox operations remain in use.

Conversation state uses `Annotated[list[dict], operator.add]`. Each node returns only its new messages. Keeping the original dictionaries preserves OpenRouter fields such as `reasoning_details` and raw tool arguments without an extra message conversion layer. This is a native LangGraph state reducer.

Each tool dispatch runs in a LangGraph `@task`. Its completed result is checkpointed separately, so a later sandbox sync failure can be retried without running that completed command again. The task accesses the graph's store through `get_store()`.

`memory/memory.py` opens `SqliteSaver` and `SqliteStore` from `langgraph-checkpoint-sqlite`, using separate connections to one SQLite file. Both stores survive process restarts. The checkpointer stores conversation state; the store holds selected facts as JSON items.

## Identity and retrieval

Two identities have different lifetimes:

| Identity | Purpose |
| --- | --- |
| Project and user | Select the durable-memory namespace shared by their conversations. |
| Thread ID | Select one conversation's checkpointed history. |

The application supplies project and user through `MemoryContext`. Tool arguments cannot choose another scope. Namespace components are hashes of these IDs; the checkpoint key hashes project, user, and the printed thread ID together, so identical thread names in different scopes do not collide. Hashes provide stable identifiers, not authentication or encryption.

The coder receives three tools: `save_memory(content)`, `search_memories(query)`, and `forget_memory(memory_id)`. The graph passes its native store and trusted runtime context to the dispatcher. Long-term memories are retrieved only when the agent calls a tool; the harness does not automatically inject memories or promote every chat message into durable facts. Conversation messages themselves are persisted as short-term checkpoints.

Recall matches any complete, case-insensitive query word against saved content. It scans native store results in pages, with no embedding service or custom ranking engine. The model tool returns up to five matches; the Python helper caps requested results at 20. Empty queries return no matches. Facts are limited to 2,000 characters, and exact repeated content reuses the same SHA-256 ID. A basic credential-pattern check rejects obvious secrets but is not a complete secret detector.

Document loaders and vector stores are separate extensions for document ingestion and semantic retrieval. Neither is required for this baseline.

## Run and resume

Follow the [README setup](../README.md#setup), configure `.env` from `.env.example`, and run `python main.py`.

| Setting | Default |
| --- | --- |
| `MEMORY_DB_PATH` | `~/.local/share/cos30018/memory.sqlite` |
| `MEMORY_PROJECT_ID` | Selected project's resolved absolute path |
| `MEMORY_USER_ID` | `default` |
| `MEMORY_THREAD_ID` | A new generated conversation ID |

The database defaults outside the project so normal project-to-sandbox copying does not include memory. Keep a custom database path outside the selected project too. Moving a project changes its default identity; set a stable `MEMORY_PROJECT_ID` if memory should follow it.

A short walkthrough:

```text
/remember This project uses pytest.
/memories
/new
Which test runner did we choose? Check your saved memories.
/resume <earlier-thread-id>
/forget <memory-id-from-memories>
```

Keep the printed thread ID to resume later. `/resume <thread-id>` selects it in the current process; setting `MEMORY_THREAD_ID` resumes it at startup. Both require the same project and user identity. `/new` gives an empty conversation while retaining that scope's durable facts. Resume restores graph history, not an old sandbox pod; each launch creates a sandbox from the selected host project.

If a model request or graph step fails, `/retry` continues the unfinished turn. The CLI blocks new messages on a thread with pending work until it is retried or a new thread is selected.

## Limits and compatibility

- SQLite is intended here for one local application. Concurrent deployments need a suitable backend and concurrency design.
- Completed tool tasks are reused on retry. A crash between an external side effect and saving its task result can still repeat that effect; there is no exactly-once execution or automatic rollback. Checkpoints do not restore unsynced files from a lost sandbox.
- LangGraph's default recursion limit bounds each invocation. If a long task reaches it, inspect the progress and use `/retry` to continue the pending turn.
- Conversation history grows without trimming or summarization. Use `/new` to start a fresh context; add a bounded-history strategy when needed.
- `/forget` deletes the durable store item only. Existing checkpoint history can still contain its content; this command is not complete privacy erasure.
- The legacy `memory/memories.db` remains untouched, is not queried by v3, and has no automatic migration. Re-save any facts needed in v3.
- Planner, reviewer, executor, and multi-agent orchestration remain incomplete and outside this memory change.

Run `python -m unittest discover -s tests -v` for local persistence, scope, tool, and graph checks. These tests use temporary databases and mocked external calls, without OpenRouter or Minikube.

## References

- [Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [LangGraph stores](https://docs.langchain.com/oss/python/langgraph/stores)
- [Checkpointer integrations](https://docs.langchain.com/oss/python/integrations/checkpointers)
- [Native SQLite store implementation](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-sqlite/langgraph/store/sqlite/base.py)
- [Document loaders](https://docs.langchain.com/oss/python/integrations/document_loaders)
- [Vector stores](https://docs.langchain.com/oss/python/integrations/vectorstores)
