import json
from typing import Any

from agent_roles.openrouter import chat_completion

MAX_TASKS = 8

PLANNER_PROMPT = """You are the planning agent for a coding assistant.
Decide whether the user's latest request needs project work. If it does, split it
into a short ordered list of independently understandable tasks. Keep tasks
small, include dependencies and likely relative file paths when known, and do
not invent files or claim work has already happened. A single task is best for
small requests. For questions that need no project edits or inspection, return
an empty tasks list and put a direct answer in summary.

Available skills are routing metadata, not evidence that a skill has been read.
When a skill fits a task, include its name and ask the worker to read it in that
task's description before coding. Preserve any skill explicitly requested by the
user (including $skill-name). You have no tools; delegate requests to inspect skill
contents to a worker instead of inventing their contents. For UI design tasks,
include frontend-design when it is available in the catalog.

Return only a JSON object with this shape:
{
  "summary": "brief plan or direct answer",
  "tasks": [
    {"id": "task-1", "title": "short title", "description": "clear deliverable",
     "depends_on": [], "files": []}
  ]
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


def create_plan(request: str, history: list[dict[str, Any]], *, skill_catalog: str = "") -> dict[str, Any]:
    """Ask the planner for a validated, bounded task list."""
    messages = [{"role": "system", "content": PLANNER_PROMPT + skill_catalog}]
    messages.extend(_text_messages(history))
    # Ensure the exact current request is present even if a long history was trimmed.
    if not messages or messages[-1].get("content") != request:
        messages.append({"role": "user", "content": request})

    reply = chat_completion(
        messages,
        model_env="OPENROUTER_PLANNER_MODEL",
        response_format={"type": "json_object"},
    )
    content = reply.get("content")
    if not isinstance(content, str):
        raise ValueError("Planner returned no JSON content.")
    plan = json.loads(content)
    if not isinstance(plan, dict):
        raise ValueError("Planner output must be a JSON object.")

    summary = plan.get("summary", "")
    if not isinstance(summary, str):
        raise ValueError("Planner summary must be text.")
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
        tasks.append({
            "id": task_id,
            "title": title.strip(),
            "description": description.strip(),
            "depends_on": dependencies,
            "files": files,
        })

    return {"summary": summary.strip(), "tasks": tasks}
