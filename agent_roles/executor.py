import json
import os
from typing import Any, Callable

from agent_roles.coder import SYSTEM_PROMPT, send_to_coder

MAX_AGENT_ROUNDS = 50


class AgentExecutionError(RuntimeError):
    """A model call failed or the agent exhausted its action budget."""


EXECUTOR_PROMPT = SYSTEM_PROMPT + """
You are a worker assigned one task by the project orchestrator. Complete only
that task, respect its file scope and dependencies, use tools to inspect and
change the project, and report what you actually changed. Do not claim tests
passed unless you ran them and saw a successful result.
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
            return {"text": result.get("text", ""), "tool_count": total_tool_count}
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
        try:
            report = json.loads(result["text"])
            if (
                not isinstance(report, dict)
                or report.get("status") not in {"complete", "incomplete", "failed"}
                or not isinstance(report.get("summary"), str)
                or not report["summary"].strip()
            ):
                raise ValueError("Invalid worker status or summary")
        except (TypeError, ValueError):
            report = {
                "status": "incomplete",
                "summary": "Worker returned an invalid task report: " + str(result["text"]),
            }
    return {
        "task_id": task["id"],
        "title": task["title"],
        "status": report["status"],
        "summary": report["summary"],
    }
