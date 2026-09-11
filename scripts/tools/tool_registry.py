"""Model-facing tools bound to a harness-selected sandbox."""
import json
from pathlib import PurePosixPath
from .terminal_use import WORKSPACE, OUTPUT_LIMIT, execute_command

SPECS = {
    'run_command': ('Run a Linux shell command ; 30 second timeout.', ['command']),
    'read_file': ('Read a  file.', ['path']),
    'write_file': ('Write a  file. Create parent directories with run_command first.', ['path', 'content']),
    'list_files': ("List a project directory", ['path']),
}


def list_tools():
    return [{'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': {key: {'type': 'string'} for key in fields},
                       'required': fields, 'additionalProperties': False},
    }} for name, (description, fields) in SPECS.items()]


def project_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError("Use a relative project path without '..'.")
    return str(PurePosixPath(WORKSPACE) / path)


def execute_tool(tool_call, *, sandbox):
    try:
        function = tool_call['function']
        name = function['name']
        if name not in SPECS:
            raise ValueError('execute_tool: unknown tool')
        arguments = json.loads(function['arguments'])
        fields = SPECS[name][1]
        if (not isinstance(arguments, dict) or set(arguments) != set(fields)
                or any(not isinstance(v, str) for v in arguments.values())):
            raise ValueError('execute_tool: invalid arguments')
        if name == 'run_command':
            result = execute_command(arguments, sandbox)
        else:
            path = project_path(arguments['path'])
            if name == 'read_file':
                content = sandbox.read_file(path).decode('utf-8')
                result = {'content': content[:OUTPUT_LIMIT], 'truncated': len(content) > OUTPUT_LIMIT}
            elif name == 'write_file':
                # Paths are anchored by project_path; preserve the absolute sandbox path.
                sandbox.write_file(path, arguments['content'])
                result = {'written': arguments['path']}
            else:
                entries = sandbox.list_files(path)
                result = {'entries': [{'name': e.name, 'type': e.type, 'size': e.size}
                                      for e in entries[:200]], 'truncated': len(entries) > 200}
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({'error': f'execute_tool: {exc}'})
