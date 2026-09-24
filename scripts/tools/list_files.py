"""List files inside the sandbox."""
import json
from .file_paths import project_path

TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List a project directory.",
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


def list_files(arguments, sandbox):
    path = project_path(arguments['path'])
    script = ('import json,pathlib,sys; '
              'print(json.dumps([dict(name=p.name,type="directory" if p.is_dir() else "file",'
              'size=p.stat().st_size) for p in pathlib.Path(sys.argv[1]).iterdir()]))')
    result = sandbox._execute(['python3', '-c', script, path])
    if result.exit_code != 0:
        raise RuntimeError('list_files failed')
    entries = json.loads(result.stdout)
    return {'entries': entries[:200], 'truncated': len(entries) > 200}
