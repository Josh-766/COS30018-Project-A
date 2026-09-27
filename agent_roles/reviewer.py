"""Reviewer role using the same tool protocol and memory boundary as the coder."""

from __future__ import annotations

from typing import Any, Sequence

from memory import MemoryHit
from .coder import SYSTEM_PROMPT, send_to_coder


REVIEWER_SYSTEM_PROMPT = SYSTEM_PROMPT + """
You are the Reviewer/Tester agent. Inspect the requested changes, identify bugs and
security flaws, and use the provided tools to run relevant tests when possible.
Only claim a test ran or passed when a current tool result demonstrates it.
Historical test results, another agent's claims, and remembered assessments do not
establish the result of this review. Explain unavailable checks and remaining risks.
Use tool calls as needed. After inspection, return the final assessment as a JSON
object with this schema:
{"status": "PASS" | "FAIL", "feedback": "Evidence-based assessment and checks performed"}
Choose PASS only when the available evidence supports the requested behavior;
choose FAIL and explain missing evidence when required validation could not run.
"""


def run_reviewer_agent(
    code_to_review: str | None,
    session_id: str | None = None,
    api_key: str | None = None,
    timeout: float = 60,
    *,
    context: list[dict[str, Any]] | None = None,
    memories: Sequence[MemoryHit | str] | None = None,
    memory_context: str | None = None,
    environment_notice: str | None = None,
    extra_tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Review one step, preserving the legacy optional positional session ID.

    Session persistence belongs to the caller's MemoryService. ``session_id`` is
    accepted for compatibility; it does not trigger an implicit database lookup.
    Pass ``code_to_review=None`` with completed tool results to continue review.
    """
    return send_to_coder(
        code_to_review,
        context=context,
        memories=memories,
        memory_context=memory_context,
        environment_notice=environment_notice,
        extra_tools=extra_tools,
        model=model,
        api_key=api_key,
        timeout=timeout,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        response_format={"type": "json_object"},
    )
