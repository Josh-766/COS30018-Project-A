
import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"

SYSTEM_PROMPT = "You are a coding agent, you look to write code that will then be reviewed and executed by another agent. Use the context, if you require more information than is available call the search memories, if not found use a tool that may be"

def send_to_coder(
    text: str,
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
    if memories:
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
            "reasoning": {"enabled": True},
        },
        timeout=timeout,
    )

    assistant_message = response.json()["choices"][0]["message"]

    updated_context = [
        *[dict(message) for message in (context or [])],
        {"role": "user", "content": text},
        dict(assistant_message),
    ]
    return {
        "text": assistant_message.get("content") or "",
        "context": updated_context,
        "message": dict(assistant_message),
    }
