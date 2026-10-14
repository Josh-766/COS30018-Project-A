import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model_env: str,
    response_format: dict[str, str] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    # Call OpenRouter using a role-specific model with a shared fallback.
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env before running agents.")

    model_kwargs = {"response_format": response_format} if response_format else {}
    client = ChatOpenAI(
        model=os.getenv(model_env) or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL,
        api_key=api_key,
        base_url=API_URL.removesuffix("/chat/completions"),
        timeout=timeout,
        max_retries=2,
        model_kwargs=model_kwargs,
    )
    response = client.invoke(messages)
    return {"content": response.content}
