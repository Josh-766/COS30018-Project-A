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


def _valid_arguments(arguments, parameters):
    """Validate declared required/optional fields without accepting bool as int."""
    if not isinstance(arguments, dict):
        return False
    properties = parameters.get('properties', {})
    if (not set(parameters.get('required', ())) <= set(arguments)
            or (parameters.get('additionalProperties') is False and not set(arguments) <= set(properties))):
        return False
    types = {'string': str, 'integer': int, 'number': (int, float),
             'boolean': bool, 'object': dict, 'array': list}
    for key, value in arguments.items():
        spec = properties.get(key, {})
        expected = spec.get('type')
        if expected in types and (not isinstance(value, types[expected])
                                  or (expected in {'integer', 'number'} and isinstance(value, bool))):
            return False
        if 'minimum' in spec and value < spec['minimum']:
            return False
        if 'maximum' in spec and value > spec['maximum']:
            return False
    return True


def execute_tool(tool_call, *, sandbox):
    try:
        function = tool_call['function']
        name = function['name']
        if name not in SPECS:
            raise ValueError('execute_tool: unknown tool')
        arguments = json.loads(function['arguments'])
        parameters = SPECS[name][0]['function']['parameters']
        if not _valid_arguments(arguments, parameters):
            raise ValueError('execute_tool: invalid arguments')
        result = SPECS[name][1](arguments, sandbox)
        return json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    except Exception as exc:
        return json.dumps({'error': f'execute_tool: {exc}'})
