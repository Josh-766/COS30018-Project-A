def response_passer(message, finish_reason=None):
    if message.get("tool_calls"):
        return "tool", message["tool_calls"]

    if message.get("content"):
        return "text", message["content"]

    if finish_reason == "length":
        return "incomplete", None

    return "empty", None