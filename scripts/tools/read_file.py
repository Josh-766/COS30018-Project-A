"""Read a file inside the sandbox."""
from .file_paths import project_path
from .terminal_use import OUTPUT_LIMIT

TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                }
            },
            "required": [
                "path"
            ],
            "additionalProperties": False
        }
    }
}


def read_file(arguments, sandbox):
    path = project_path(arguments['path'])
    result = sandbox._execute(['cat', '--', path])
    if result.exit_code != 0:
        raise RuntimeError('read_file failed')
    content = result.stdout.decode('utf-8')
    return {'content': content[:OUTPUT_LIMIT], 'truncated': len(content) > OUTPUT_LIMIT}
