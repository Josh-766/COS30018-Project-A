# COS30018 Project A

A coding-agent harness with an OpenRouter planner, sequential task workers, a reviewing agent, file and terminal tools, and a Minikube sandbox. LangGraph owns the workflow, conversation checkpoints, and durable project memories.

## Setup

Use Python 3.10 or newer, with [Minikube](https://minikube.sigs.k8s.io/docs/start/) and [kubectl](https://kubernetes.io/docs/tasks/tools/) installed and a working Minikube driver.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

In Windows PowerShell, use:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set `OPENROUTER_API_KEY` in `.env`. Choose a tool-capable model through `OPENROUTER_MODEL`; `openrouter/free` is the fallback. Then run:

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

## Coding skills

Skills are reusable instructions for this project's own agents. They are separate
from saved memories and do not require Claude Code or Codex. Two original Anthropic
skills are bundled in `skills/`:

- [frontend-design](skills/frontend-design/SKILL.md): visual direction, typography,
  layout, responsive interfaces and design critique.
- [webapp-testing](skills/webapp-testing/SKILL.md): Playwright workflows for local
  web applications, with a server helper and examples.

See [sources and licenses](skills/SOURCES.md) for the pinned upstream revision and
dependency details. Install the updated `requirements.txt` for the YAML parser.

The single agent, planner, workers and reviewer receive a small catalog of skill
names and descriptions. Coding agents are instructed to call `read_skill` before
using a relevant skill; full instructions are returned as a tool result only when
requested. `list_skills` refreshes the catalog and `read_skill_resource` reads linked
text resources. The planner delegates skill use to workers. Selection is performed
by the model, so you can name a skill explicitly when you want it used:

```text
Use $frontend-design to build a responsive landing page for a Melbourne bakery.
Use $webapp-testing to check the signup form and report any failures.
```

| Command | Effect |
| --- | --- |
| `/skills` | List available skills, their descriptions and any invalid entries. |
| `/skill frontend-design` | Print a skill's instructions and resource names. |

These CLI commands only inspect files; they do not call the model or select a skill
for the next request. Use a normal chat request such as the examples above to apply it.
Skill tool calls appear in the existing tool output and multi-agent observation logs.

To add another skill, place its folder in this harness's `skills/` directory:

```text
skills/
  my-skill/
    SKILL.md
    references/       # optional
    scripts/          # optional
```

The minimum `SKILL.md` is:

```markdown
---
name: my-skill
description: Explain what this skill does and when the agent should use it.
---
Write the task-specific instructions here. Link supporting files using paths
relative to this folder, for example references/example.md.
```

Names must match the folder, use lowercase letters/digits/hyphens, and be at most
64 characters; descriptions are limited to 1024 characters. UTF-8 files up to
128 KiB are supported. Invalid entries are reported without hiding other skills.
Copy the complete upstream folder, including referenced files and its license.
New or edited skills are discovered on the next catalog/tool read; no restart is needed.

When you select another project at startup, its `skills/` directory is also read.
A valid project skill overrides a bundled skill with the same name for that session.
Bundled skills remain available when the selected project has none. The loader does
not consult personal Claude/Codex directories and does not put skills in the memory DB.

Skill resources are read from the host by the skill tools, while project commands
still run in Minikube. If a helper is needed, the worker can read its text with
`read_skill_resource`, copy it into the sandbox using `write_file`, and run it there.
Loading a skill never executes its scripts or installs dependencies. Browser tests
need Playwright and a browser installed in the sandbox; the default Python image
does not provide them. This harness has no image-viewing tool, so agents can inspect
DOM/test results and capture screenshots when dependencies exist, but cannot claim
to have visually reviewed the screenshots. Skill guidance remains subordinate to
the user's request, the agent's role and the tools actually available.

## Tests

The memory and graph tests use temporary databases and mocked model/sandbox calls; they require no API credentials, external API requests, or running Minikube.

```sh
python -m unittest discover -s tests -v
```

Sandbox background: [coding agents](https://agent-sandbox.sigs.k8s.io/docs/use-cases/coding-agents/) and [LangChain integration](https://agent-sandbox.sigs.k8s.io/docs/use-cases/examples/langchain/).
