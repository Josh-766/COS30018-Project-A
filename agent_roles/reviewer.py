import json
import os
from typing import Any, Callable

from agent_roles.coder import SYSTEM_PROMPT
from agent_roles.executor import run_agent
from scripts.tools.tool_registry import SKILL_SPECS, list_tools

REVIEWER_PROMPT = """
You are the independent reviewer for and tester of software. Inspect the relevant project files,
check the code against the initial request and requirements.

The conversation history contains the same original request messages the planner saw.
The review packet that follows contains the plan and worker reports. Treat the
initial request as authoritative if it conflicts with the plan or its review notes.

Verification policy:
- Validate the implementation against the initial request, the plan's requirements,
  and the worker reports. Do not treat planner notes as requirements unless the
  initial request supports them.

If the requirements are met, the project can PASS.

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
    planner_notes: list[str] | None = None,
) -> dict[str, str]:
    payload = {
        "original_request": request,
        "planned_tasks": tasks,
        "planner_notes": planner_notes or [],
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
