# Observation

All observation code is in `observation/observation.py`. It edits no other file: `install()` wraps functions the graphs already call, and the two factories return the normal graphs with those hooks on.

## Turning it on

In `main.py`, swap one import:

```python
# from agent_roles.graph import create_orchestrator_graph
from observation import create_observed_orchestrator as create_orchestrator_graph
```

For the single-agent baseline, use `create_observed_coder_graph` in place of `create_coder_graph`. Both write the same trace format, so the metrics compare directly.

## What it hooks

| Wrapped | Why |
| --- | --- |
| `graph.execute_tool` | Every tool call by workers, reviewer or baseline coder becomes an Observation |
| `executor.send_to_coder`, `graph.send_to_coder` | LLM latency and tokens; ends a stuck agent's loop |
| `graph.create_plan` | Planner latency, tokens, task count |
| `graph.execute_task` | Worker scope; stuck or round-limit → `status: "failed"` task result |
| `graph.review_work` | Reviewer scope; stuck or round-limit → `FAIL` review |
| `coder.requests`, `openrouter.ChatOpenAI` | Read token usage from responses |
| the sandbox (`ObservedSandbox`) | Error details, head+tail output, binary-safe reads, skip syncs after reads |
| the compiled graph (`ObservedGraph`) | `turn_start` / `turn_end` around each request |

`uninstall()` restores everything (used by tests).

## What agents see

Sandbox tool results keep their fields, with three added at the front:

```json
{"ok": false, "error_type": "test_fail", "summary": "FAILED (failures=1)", "stdout": "...", "exit_code": 1}
```

`error_type`: `syntax`, `import`, `command_not_found`, `test_fail`, `permission`, `not_found`, `exception`, `timeout`, `nonzero_exit`, `bad_call`, `bad_path`, `tool_error`.

## Failure handling

Within one agent run (one worker task, or one review):

- the same failing call 3 times adds an `observer_note` to the tool result telling the agent to change approach;
- 5 times stops the agent. The observer answers its next model call itself, so `run_agent` ends normally.

A stopped worker, or one that hits `run_agent`'s 12-round limit, is returned to the blackboard as a **failed** task with the reason, instead of crashing the run. The reviewer sees it and the orchestrator's repair loop handles it. A stopped reviewer becomes a `FAIL` review. Other errors (API failures, bad planner JSON) still propagate so `/retry` works as before.

## Traces and metrics

One JSONL file per conversation in `~/.local/share/cos30018/traces` (set `TRACE_DIR` to change).

Events: `turn_start`, `plan`, `llm_call`, `tool`, `stuck_nudge`, `stuck_stop`, `task_end`, `review`, `turn_end`. Each carries `agent` (planner / worker / reviewer / coder) and `task_id` where relevant.

```sh
python -m observation                  # summary of all traces
python -m observation --csv rows.csv   # one row per user request
python -m observation --json
```

Reported: request outcomes (completed / failed / answered / error), success rate, first-review pass rate, mean reviews, worker tasks failed, tool success rate, recovery rate (a failed tool later succeeds in the same agent run), task recovery rate (a request with a failure still completes), observer nudges and stops, mean LLM calls, median request time, tokens, errors by type, and a per-agent breakdown.

## Limits

- Hooks replace module attributes, so a rename of a wrapped function in `graph.py`, `executor.py` or `coder.py` needs a matching change here. The tests catch this.
- Classification is regex-based; unknown output falls back to `nonzero_exit`.
- Trace files are named by the hashed thread key, not the printed thread ID.
- Traces may contain task text and file contents; they live outside the project and are not removed by `/forget`.
