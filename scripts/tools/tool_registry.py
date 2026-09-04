import json

from .terminal_use import POWERSHELL_TOOL, execute_powershell


TOOLS = {
    "powershell": {
        "definition": POWERSHELL_TOOL,
        "execute": execute_powershell,
    }
}


def list_tools():
    return [tool["definition"] for tool in TOOLS.values()]


def execute_tool(tool_call):
    function = tool_call["function"]
    tool = TOOLS.get(function["name"])

    if tool is None:
        return f"Unknown tool: {function['name']}"

    arguments = json.loads(function["arguments"])
    return tool["execute"](arguments)
