import json
import os
from typing import Any, Callable

from agent_roles.coder import SYSTEM_PROMPT, send_to_coder

MAX_AGENT_ROUNDS = 100


class AgentExecutionError(RuntimeError):
    """A model call failed or the agent exhausted its action budget."""


EXECUTOR_PROMPT =  """
You are a coding worker agent assigned one task by the project orchestrator. Complete only
that task, respect its file scope and dependencies, use tools to inspect and
change the project, and report what you actually changed. Keep changes focused on your task. Don't run or write excessive tests. Be efficient with tool calls to complete the task.

Verification policy:
- Use only tools and packages already present in the sandbox.
- Never run pip/npm/apt installs, download browsers, or fetch system packages.
- If a check needs something unavailable, skip it and report it as a blocker.
- Run at most one focused check after each edit. Do not rerun a check that
  already passed unless you changed the code it covers.
- Prefer the task's listed checks. If none are listed, use the smallest
  syntax/import check for the files you changed.
- Batch related checks into one command, and keep commands under 30 seconds.

If you have learnings that are core for the next worker you can write them to a project_learnings.md file so the next worker/workers can find them. 

Return your final response as a JSON object without markdown, with keys:
{"status": "complete" or "incomplete" or "failed", "summary": "changes, checks
actually run, and remaining blockers"}.
Use incomplete when work remains, failed when you cannot perform the task, and
complete only when the assigned task is finished."""


def _emit(on_event, **event):
    if on_event is not None:
        on_event(event)


def _reasoning_text(message):
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts = [
            item.get("text") for item in details
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def run_agent(
    prompt: str,
    *,
    history: list[dict[str, Any]],
    system_prompt: str,
    model: str | None,
    run_tool: Callable[[dict[str, Any]], str],
    tool_definitions: list[dict[str, Any]] | None = None,
    max_rounds: int = MAX_AGENT_ROUNDS,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    agent: str = "agent",
    task_id: str | None = None,
) -> dict[str, Any]:
    # Run one tool-capable role until it returns a final answer.
    context = [dict(message) for message in history]
    try:
        result = send_to_coder(
            prompt,
            context=context,
            model=model,
            system_prompt=system_prompt,
            tool_definitions=tool_definitions,
        )
    except Exception as error:
        raise AgentExecutionError(f"Model call failed: {error}") from error
    # Keep the role's task contract after a skill read (and all other tool calls).
    # send_to_coder builds its own request list; it does not mutate this history.
    context.append({"role": "user", "content": prompt})

    total_tool_count = 0
    for round_index in range(max_rounds + 1):
        message = result.get("message", {})
        thinking = _reasoning_text(message)
        if thinking:
            _emit(on_event, agent=agent, task_id=task_id,
                  type="thinking", text=thinking)

        tool_calls = result.get("tool_calls", [])
        if not tool_calls:
            return {"text": result.get("text", ""), "tool_count": total_tool_count,
                    "context": context}
        if round_index == max_rounds:
            break

        context.append(dict(message))
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "?")
            arguments = function.get("arguments", "")
            _emit(on_event, agent=agent, task_id=task_id,
                  type="tool_call", tool=name, arguments=arguments)
            output = run_tool(call)
            context.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": output,
            })
            _emit(on_event, agent=agent, task_id=task_id,
                  type="tool_result", tool=name, content=output)
            total_tool_count += 1

        try:
            result = send_to_coder(
                None,
                context=context,
                model=model,
                system_prompt=system_prompt,
                tool_definitions=tool_definitions,
            )
        except Exception as error:
            raise AgentExecutionError(f"Model call failed: {error}") from error

    raise AgentExecutionError(f"Agent exceeded its {max_rounds}-round tool limit.")


def execute_task(
    task: dict[str, Any],
    *,
    request: str,
    history: list[dict[str, Any]],
    previous_results: list[dict[str, Any]],
    run_tool: Callable[[dict[str, Any]], str],
    review_feedback: str = "",
    on_event: Callable[[dict[str, Any]], None] | None = None,
    skill_catalog: str = "",
) -> dict[str, Any]:
    # Execute one task contract and return a compact result for the blackboard.
    model = os.getenv("OPENROUTER_EXECUTOR_MODEL")
    prompt = {
        "user_request": request,
        "task": task,
        "completed_tasks": previous_results,
        "review_feedback_to_address": review_feedback,
        "instructions": (
            "Work on this task in the shared project sandbox. Read existing code first. "
            "Keep changes within the task's listed files when possible. Coordinate with "
            "earlier task results. Run focused verification if practical and report exact outcomes."
        ),
    }
    _emit(
        on_event,
        agent="worker",
        task_id=task["id"],
        type="task_start",
        title=task.get("title", ""),
        description=task.get("description", ""),
        files=task.get("files", []),
        depends_on=task.get("depends_on", []),
        context=task.get("context", []),
    )
    try:
        result = run_agent(
            json.dumps(prompt, ensure_ascii=False),
            history=history,
            system_prompt=EXECUTOR_PROMPT + skill_catalog,
            model=model,
            run_tool=run_tool,
            on_event=on_event,
            agent="worker",
            task_id=task["id"],
        )
    except AgentExecutionError as error:
        report = {"status": "failed", "summary": str(error)}
    else:
        text = result.get("text", "")
        context = result.get("context") or history
        report = None
        for attempt in range(2):
            try:
                start = (text or "").find("{")
                if start < 0:
                    raise ValueError("No task report found")
                parsed, _ = json.JSONDecoder().raw_decode(text[start:])
                if (
                    not isinstance(parsed, dict)
                    or parsed.get("status") not in {"complete", "incomplete", "failed"}
                    or not isinstance(parsed.get("summary"), str)
                    or not parsed["summary"].strip()
                ):
                    raise ValueError("Invalid task report")
                report = {"status": parsed["status"], "summary": parsed["summary"]}
                break
            except (TypeError, ValueError):
                if attempt:
                    break
                correction = (
                    "Your previous response was:\n"
                    + (text or "")[:4000]
                    + "\n\nThat response did not contain a valid task report. "
                    "Return only a JSON object with exactly these keys: "
                    '{"status": "complete", "summary": "changes, checks actually run, '
                    'and remaining blockers"}. The status must be one of "complete", '
                    '"incomplete", or "failed". Use complete only when the task is '
                    "finished, incomplete when work remains, and failed when you could "
                    "not perform the task. Do not use markdown or code fences."
                )
                try:
                    corrected = send_to_coder(
                        correction,
                        context=context,
                        model=model,
                        system_prompt=EXECUTOR_PROMPT + skill_catalog,
                        tool_definitions=[],
                    )
                    text = corrected.get("text", "")
                except Exception as error:
                    report = {
                        "status": "incomplete",
                        "summary": (
                            "Worker returned an invalid task report and the report "
                            "retry failed: " + str(error)
                        ),
                    }
                    break
        if report is None:
            report = {
                "status": "incomplete",
                "summary": "Worker returned an invalid task report: " + str(text),
            }
    return {
        "task_id": task["id"],
        "title": task["title"],
        "status": report["status"],
        "summary": report["summary"],
    }
