"""Write a file inside the sandbox."""
from .file_paths import project_path

TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file, creating parent directories if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "content": {
                    "type": "string"
                }
            },
            "required": [
                "path",
                "content"
            ],
            "additionalProperties": False
        }
    }
}


def write_file(arguments, sandbox):
    path = project_path(arguments['path'])
    content = arguments['content'].encode('utf-8')
    result = sandbox._execute(
        ['sh', '-c', 'mkdir -p "$(dirname "$1")" && cat > "$1"', 'sh', path],
        input_data=content,
    )
    if result.exit_code != 0:
        raise RuntimeError('write_file failed')
    return {'written': arguments['path']}
