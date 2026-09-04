import subprocess

POWERSHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "powershell",
        "description": "Run a PowerShell command in the terminal.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run."}
            },
            "required": ["command"],
        },
    },
}
def execute_powershell(arguments):
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", arguments["command"]],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (result.stdout + result.stderr).strip()
