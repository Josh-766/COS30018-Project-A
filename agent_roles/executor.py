import json
import os
from typing import Any, Callable

from agent_roles.coder import SYSTEM_PROMPT, send_to_coder

MAX_AGENT_ROUNDS = 12

EXECUTOR_PROMPT = SYSTEM_PROMPT + """
You are a worker assigned one task by the project orchestrator. Complete only
that task, respect its file scope and dependencies, use tools to inspect and
change the project, and report what you actually changed. Do not claim tests
passed unless you ran them and saw a successful result."""


def run_agent(
    prompt: str,
    *,
    history: list[dict[str, Any]],
    system_prompt: str,
    model: str | None,
    run_tool: Callable[[dict[str, Any]], str],
    tool_definitions: list[dict[str, Any]] | None = None,
    max_rounds: int = MAX_AGENT_ROUNDS,
) -> dict[str, Any]:
    # Run one tool-capable role until it returns a final answer.
    context = [dict(message) for message in history]
    result = send_to_coder(
        prompt,
        context=context,
        model=model,
        system_prompt=system_prompt,
        tool_definitions=tool_definitions,
    )

    total_tool_count = 0
    for _ in range(max_rounds):
        tool_calls = result.get("tool_calls", [])
        if not tool_calls:
            return {"text": result.get("text", ""), "tool_count": total_tool_count}

        context.append(dict(result["message"]))
        for call in tool_calls:
            output = run_tool(call)
            context.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": output,
            })
            total_tool_count += 1
        result = send_to_coder(
            None,
            context=context,
            model=model,
            system_prompt=system_prompt,
            tool_definitions=tool_definitions,
        )

    raise RuntimeError(f"Agent exceeded its {max_rounds}-round tool limit.")


def execute_task(
    task: dict[str, Any],
    *,
    request: str,
    history: list[dict[str, Any]],
    previous_results: list[dict[str, Any]],
    run_tool: Callable[[dict[str, Any]], str],
    review_feedback: str = "",
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
    result = run_agent(
        json.dumps(prompt, ensure_ascii=False),
        history=history,
        system_prompt=EXECUTOR_PROMPT,
        model=model,
        run_tool=run_tool,
    )
    return {
        "task_id": task["id"],
        "title": task["title"],
        "status": "complete",
        "summary": result["text"],
    }
