import json
import os
from typing import Any, Callable

from agent_roles.coder import send_to_coder
from scripts.tools.tool_registry import list_tools

MAX_TASKS = 8

PLANNER_PROMPT = """You are the planning agent for a coding assistant.
Decide whether the user's latest request needs project work. If it does, split it
into a short ordered list of independently understandable tasks. Keep tasks
small, include dependencies and likely relative file paths when known, and do
not invent files or claim work has already happened. A single task is best for
small requests. For questions that need no project edits or inspection, return
an empty tasks list and put a direct answer in summary.

You have read-only project and memory tools; use them when inspection is needed.

Available skills are routing metadata, not evidence that a skill has been read.
When a skill fits a task, include its name in that task's description and context,
and ask the worker to read it before coding. Preserve any skill explicitly
requested by the user (including $skill-name). For UI design
tasks, include frontend-design when it is available in the catalog.

For each task, include a "context" array with specific, concise facts the worker
needs. Be clear: the worker sees only this task contract and context, not the
conversation history. Also include a top-level "review_context" array with the key
requirements for the reviewer.

Return only a JSON object with this shape:
{
  "summary": "brief plan or direct answer",
  "tasks": [
    {"id": "task-1", "title": "short title", "description": "clear deliverable",
     "depends_on": [], "files": [], "context": ["relevant fact 1", "relevant fact 2"]}
  ],
  "review_context": ["key requirement", "design decision"]
}
Use at most 8 tasks. Tasks run in the order returned. Do not include markdown."""


def _text_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    result = []
    for message in history[-8:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            result.append({"role": role, "content": content[-6000:]})
    return result


def create_plan(
    request: str,
    history: list[dict[str, Any]],
    *,
    run_tool: Callable[[dict[str, Any]], str],
    skill_catalog: str = "",
) -> dict[str, Any]:
    """Ask the planner for a validated, bounded task list.
    The planner has read-only project and memory tools to inspect the project."""
    # send_to_coder prepends the system prompt itself, so keep it out of messages.
    messages = list(_text_messages(history))
    # Ensure the exact current request is present even if a long history was trimmed.
    if not messages or messages[-1].get("content") != request:
        messages.append({"role": "user", "content": request})

    tool_defs = [
        tool for tool in list_tools()
        if tool["function"]["name"] in {"read_file", "list_files",
                                         "save_memory", "search_memories", "forget_memory"}
    ]

    for _ in range(6):
        result = send_to_coder(
            None, context=messages, model=os.getenv("OPENROUTER_PLANNER_MODEL"),
            system_prompt=PLANNER_PROMPT + skill_catalog,
            tool_definitions=tool_defs,
        )
        tool_calls = result.get("tool_calls", [])
        if not tool_calls:
            break
        messages.append(dict(result["message"]))
        for call in tool_calls:
            output = run_tool(call)
            messages.append({
                "role": "tool", "tool_call_id": call["id"], "content": output,
            })
    else:
        raise RuntimeError("Planner exceeded its tool-call limit.")

    content = result.get("text", "")
    if not content:
        raise ValueError("Planner returned no content.")

    #create content for the context that will be passed to the other agents
    content = content.strip()
    if content.startswith("```"):
        idx = content.find("\n")
        if idx != -1:
            content = content[idx:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

    plan = json.loads(content)
    if not isinstance(plan, dict):
        raise ValueError("Planner output must be a JSON object.")

    summary = plan.get("summary", "")
    if not isinstance(summary, str):
        raise ValueError("Planner summary must be text.")

    review_context = plan.get("review_context", [])
    if not isinstance(review_context, list) or any(
        not isinstance(item, str) for item in review_context
    ):
        review_context = []

    raw_tasks = plan.get("tasks", [])
    if not isinstance(raw_tasks, list) or len(raw_tasks) > MAX_TASKS:
        raise ValueError(f"Planner must return between 0 and {MAX_TASKS} tasks.")

    tasks = []
    seen_ids: set[str] = set()
    for index, task in enumerate(raw_tasks, 1):
        if not isinstance(task, dict):
            raise ValueError(f"Planner task {index} must be an object.")
        title = task.get("title")
        description = task.get("description")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Planner task {index} needs a title.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Planner task {index} needs a description.")
        files = task.get("files", [])
        dependencies = task.get("depends_on", [])
        if not isinstance(files, list) or any(not isinstance(item, str) for item in files):
            raise ValueError(f"Planner task {index} files must be a list of paths.")
        if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
            raise ValueError(f"Planner task {index} dependencies must be a list of task IDs.")
        task_id = str(task.get("id") or f"task-{index}").strip()
        if not task_id or task_id in seen_ids:
            raise ValueError(f"Planner task {index} needs a unique nonempty ID.")
        if any(dependency not in seen_ids for dependency in dependencies):
            raise ValueError(f"Planner task {task_id} depends on an unknown or later task.")
        seen_ids.add(task_id)

        task_context = task.get("context", [])
        if not isinstance(task_context, list) or any(
            not isinstance(item, str) for item in task_context
        ):
            task_context = []

        tasks.append({
            "id": task_id,
            "title": title.strip(),
            "description": description.strip(),
            "depends_on": dependencies,
            "files": files,
            "context": task_context,
        })

    return {
        "summary": summary.strip(),
        "tasks": tasks,
        "review_context": review_context,
    }