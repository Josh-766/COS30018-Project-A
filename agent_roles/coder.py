
import os
from typing import Any

import requests
from dotenv import load_dotenv
from response_passer import response_passer
from scripts.tools.tool_registry import list_tools


load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"

SYSTEM_PROMPT = """You are a coding agent working in a live Linux sandbox.
The project is /home/user/project. Use the supplied tools to inspect, edit, and test it.
File tools take relative project paths. Commands start in the project directory;
shell state such as cd does not persist between commands, but files do. Do not assume internet access or installed packages.
"""

def send_to_coder(
    text: str | None,
    *,
    context: list[dict[str, Any]] | None = None,
    memories: list[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 60,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:

    api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    request_messages: list[dict[str, Any]] = []
    
    request_messages.append({"role": "system", "content": system_prompt})
    request_messages.extend(dict(message) for message in (context or []))
    if text is None:
        pass
    elif memories:
        request_messages.append({"role": "user", "content": text + "\n\nRetrieved context:\n" + "\n".join(memories)})
    else:
        request_messages.append({"role": "user", "content": text})

        
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL,
            "messages": request_messages,
            "tools": list_tools(),
            "reasoning": {"enabled": True},
        },
        timeout=timeout,
    )

    response.raise_for_status()
    payload = response.json()
    model_response = payload["choices"][0]
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
