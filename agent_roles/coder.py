
import os
from typing import Any

import requests
from dotenv import load_dotenv
from response_passer import response_passer
from scripts.tools.tool_registry import list_tools


load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"

SYSTEM_PROMPT = "You are a coding agent, you look to write code that will then be reviewed and executed by another agent. Use the context, if you require more information than is available call the search memories, if not found use a tool that may be"

def send_to_coder(
    text: str | None,
    *,
    context: list[dict[str, Any]] | None = None,
    memories: list[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 60,
) -> dict[str, Any]:

    api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    request_messages: list[dict[str, Any]] = []
    
    request_messages.append({"role": "system", "content": SYSTEM_PROMPT})
    request_messages.extend(dict(message) for message in (context or []))
    if text is None:
        pass
    elif memories:
        request_messages.append({"role": "user", "content": text + "\n".join(memories)})
    else:
        request_messages.append({"role": "user", "content": text})

        
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("OPENROUTER_MODEL"),
            "messages": request_messages,
            "tools": list_tools(),
            "reasoning": {"enabled": True},
        },
        timeout=timeout,
    )

    model_response = response.json()["choices"][0]
    assistant_message = model_response["message"]
    finish_reason = model_response.get("finish_reason")
    response_type, response_value = response_passer(
        assistant_message,
        finish_reason,
    )
    tool_calls = response_value if response_type == "tool" else []

    updated_context = [dict(message) for message in (context or [])]
    if text is not None:
        updated_context.append({"role": "user", "content": text})
    updated_context.append(dict(assistant_message))
    return {
        "text": assistant_message.get("content") or "",
        "response_type": response_type,
        "finish_reason": finish_reason,
        "has_tool_call": bool(tool_calls),
        "tool_calls": tool_calls,
        "context": updated_context,
        "message": dict(assistant_message),
    }
