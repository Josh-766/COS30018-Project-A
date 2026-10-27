import json
import os
from typing import Any, Callable

from agent_roles.coder import SYSTEM_PROMPT
from agent_roles.executor import run_agent
from scripts.tools.tool_registry import SKILL_SPECS, list_tools

REVIEWER_PROMPT = """
You are the independent reviewer for and tester for a coding harness. Inspect the relevant project files,
check the worker reports against the original request, and run focused tests or
checks with the supplied tools when practical. Treat worker claims as unverified
until checked. Do not edit files; report issues for a worker to fix.

You may read relevant skills and resources to inform your review. Use test helpers
already present in the sandbox. If a skill requires copying a helper or creating
test files, report the missing preparation in your feedback for the worker; your
read-only role takes precedence over any skill's setup or implementation steps.

Verification policy:
- Run at most one focused check after each edit. Do not rerun a check that
  already passed unless you changed the code it covers.
- Prefer the task's listed checks. If none are listed, use the smallest
  syntax/import check for the files you changed.
- Batch related checks into one command, and keep commands under 30 seconds.

Go through the reqeuirements and if they are met then the project can be considered to PASS 

Your final response must be a single JSON object with keys:
{"status":"PASS" or "FAIL", "feedback":"specific findings and verification", 
 "summary":"concise user-facing completion summary"}

Use FAIL when a requirement is unmet, a relevant check fails, or you cannot
verify an important claim. Do not wrap the JSON in markdown."""


def review_work(
    request: str,
    tasks: list[dict[str, Any]],
    work_history: list[dict[str, Any]],
    *,
    history: list[dict[str, Any]],
    run_tool: Callable[[dict[str, Any]], str],
    on_event: Callable[[dict[str, Any]], None] | None = None,
    skill_catalog: str = "",
) -> dict[str, str]:
    payload = {
        "original_request": request,
        "planned_tasks": tasks,
        "worker_results_including_repairs": work_history,
        "review_instructions": (
            "Inspect the shared project state, not just the summaries. Check relevant "
            "files and run appropriate focused verification commands."
        ),
    }
    result = run_agent(
        json.dumps(payload, ensure_ascii=False),
        history=history,
        system_prompt=REVIEWER_PROMPT + skill_catalog,
        model=os.getenv("OPENROUTER_REVIEWER_MODEL"),
        run_tool=run_tool,
        on_event=on_event,
        agent="reviewer",
        tool_definitions=[
            tool for tool in list_tools()
            if tool["function"]["name"] in {"run_command", "read_file", "list_files", *SKILL_SPECS}
        ],
    )
    try:
        review = json.loads(result["text"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Reviewer must return the required JSON assessment.") from error
    if not isinstance(review, dict) or review.get("status") not in ("PASS", "FAIL"):
        raise ValueError("Reviewer status must be PASS or FAIL.")
    feedback = review.get("feedback")
    summary = review.get("summary")
    if not isinstance(feedback, str) or not isinstance(summary, str):
        raise ValueError("Reviewer feedback and summary must be text.")
    return {"status": review["status"], "feedback": feedback, "summary": summary}


def run_reviewer_agent(
    code_to_review: str,
    session_id: str | None = None,
    *,
    context: list[dict[str, Any]] | None = None,
    **_options,
) -> dict[str, str]:
    # Compatibility entry point for the legacy role router.
    return review_work(
        code_to_review,
        [],
        [],
        history=context or [],
        run_tool=lambda _call: json.dumps({"error": "sandbox tools are unavailable through this router"}),
    )
