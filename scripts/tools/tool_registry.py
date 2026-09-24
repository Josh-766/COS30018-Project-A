"""Register tool definitions and dispatch model tool calls."""
import json
from .terminal_use import TOOL as COMMAND_TOOL, execute_command
from .read_file import TOOL as READ_FILE_TOOL, read_file
from .write_file import TOOL as WRITE_FILE_TOOL, write_file
from .list_files import TOOL as LIST_FILES_TOOL, list_files

SPECS = {
    'run_command': (COMMAND_TOOL, execute_command),
    'read_file': (READ_FILE_TOOL, read_file),
    'write_file': (WRITE_FILE_TOOL, write_file),
    'list_files': (LIST_FILES_TOOL, list_files),
}


def list_tools():
    return [definition for definition, execute in SPECS.values()]


def execute_tool(tool_call, *, sandbox):
    try:
        function = tool_call['function']
        name = function['name']
        if name not in SPECS:
            raise ValueError('execute_tool: unknown tool')
        arguments = json.loads(function['arguments'])
        fields = SPECS[name][0]['function']['parameters']['required']
        if (not isinstance(arguments, dict) or set(arguments) != set(fields)
                or any(not isinstance(v, str) for v in arguments.values())):
            raise ValueError('execute_tool: invalid arguments')
        result = SPECS[name][1](arguments, sandbox)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({'error': f'execute_tool: {exc}'})
