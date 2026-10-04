# COS30018 Project A

A coding-agent harness with an OpenRouter coder, file and terminal tools, and a Minikube sandbox. The current `mem_v3` implementation uses LangGraph for conversation checkpoints and durable project memories. Planner, reviewer, executor, and multi-agent orchestration are still incomplete.

## Setup

Use Python 3.10 or newer, with [Minikube](https://minikube.sigs.k8s.io/docs/start/) and [kubectl](https://kubernetes.io/docs/tasks/tools/) installed and a working Minikube driver.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. Choose a tool-capable model through `OPENROUTER_MODEL`; the existing default is `openrouter/free`. Then run:

```sh
python main.py
```

Select the project directory when prompted. The harness starts Minikube, copies the project into its sandbox, and saves sandbox changes back to the selected directory.

## Memory

Short-term memory is conversation history persisted by LangGraph's `SqliteSaver`. Long-term memory consists of selected facts in its native `SqliteStore`, shared across conversations for the same project and user. The agent can call `search_memories`, `save_memory`, and `forget_memory` as needed.

| Command | Effect |
| --- | --- |
| `/remember <fact>` | Save a durable fact. |
| `/memories` | List saved facts and their IDs. |
| `/forget <id>` | Delete a durable fact; conversation checkpoints still retain their history. |
| `/new` | Start a fresh conversation, keeping long-term memories. |
| `/resume <thread-id>` | Select a conversation using its previously printed ID. |
| `/retry` | Continue an unfinished turn from its checkpoint. |
| `exit` or `quit` | End the session. |

Memory defaults to `~/.local/share/cos30018/memory.sqlite`, outside the project directory. The old `memory/memories.db` remains untouched and is neither queried nor migrated. See the [memory design and run guide](Documentation/memory-v3.md) for configuration, limitations, and framework references.

## Tests

The memory and graph tests use temporary databases and mocked model/sandbox calls; they require no API credentials, external API requests, or running Minikube.

```sh
python -m unittest discover -s tests -v
```

Sandbox background: [coding agents](https://agent-sandbox.sigs.k8s.io/docs/use-cases/coding-agents/) and [LangChain integration](https://agent-sandbox.sigs.k8s.io/docs/use-cases/examples/langchain/).
