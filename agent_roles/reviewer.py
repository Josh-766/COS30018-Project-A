import os
from typing import Any

import requests
from dotenv import load_dotenv

# Import sandbox and memory
# import sandbox_setup
from memory import get_relevant_memories

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Strict System Prompt
REVIEWER_SYSTEM_PROMPT = """You are a Reviewer/Tester agent in a multi-agent coding system. 
Your role is to analyze the provided code, execute tests using the provided sandbox tool, and identify bugs or security flaws.
You must return your final assessment strictly in JSON format matching this schema:
{
    "status": "PASS" | "FAIL",
    "feedback": "Detailed explanation of errors for the Coder, or a success message."
}"""


def run_reviewer_agent(
    code_to_review: str,
    session_id: str,
    api_key: str | None = None,
    timeout: float = 60,
) -> dict[str, Any]:

    api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    # Retrieve current context from memory
    context = get_context(session_id)

    request_messages: list[dict[str, Any]] = [
        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT}
    ]
    request_messages.extend(context)

    user_content = f"Please review and test the following code:\n\n{code_to_review}"
    request_messages.append({"role": "user", "content": user_content})

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
                "messages": request_messages,
                "tools": [SANDBOX_TOOL],  # Will be imported from tools
                "response_format": {"type": "json_object"},
                "reasoning": {"enabled": True},
            },
            timeout=timeout,
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        return {"error": f"API Request failed: {str(e)}", "status": "ERROR"}

    assistant_message = response.json()["choices"][0]["message"]
    tool_calls = assistant_message.get("tool_calls", [])

    # Update Memory
    append_to_context(session_id, {"role": "user", "content": user_content})
    append_to_context(session_id, dict(assistant_message))

    return {
        "text": assistant_message.get("content") or "",
        "tool_calls": tool_calls,
        "message": dict(assistant_message),
    }
