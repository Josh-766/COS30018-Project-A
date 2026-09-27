"""Read a recoverable page of source code inside the sandbox."""
import json
import inspect

from memory import privacy

from .file_paths import project_path


# Leave room for page metadata inside the agent's 4,000-character observation.
# This is an encoded JSON budget, so quotes/control characters count accurately.
CONTENT_BUDGET = 3000
DEFAULT_MAX_LINES = 200
MAX_PAGE_LINES = 2000
MAX_SOURCE_BYTES = 2 * 1024 * 1024
# Execute our trusted standalone filter in the sandbox, without relying on the
# target project containing or importing the agent's Python package.
_PRIVACY_SOURCE = inspect.getsource(privacy)

TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file page (maximum file size 2 MiB). Follow next_line/next_column when truncated. Credentials are masked; redacted=true means content differs from the file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "start_line": {
                    "type": "integer", "minimum": 1,
                    "description": "First line to read (1-based, default 1)."
                },
                "max_lines": {
                    "type": "integer", "minimum": 1, "maximum": MAX_PAGE_LINES,
                    "description": "Maximum lines to read (default 200); output also has a character budget."
                },
                "start_column": {
                    "type": "integer", "minimum": 1,
                    "description": "First character on start_line (1-based, default 1). Use next_column for a long line."
                }
            },
            "required": [
                "path"
            ],
            "additionalProperties": False
        }
    }
}


_PAGE_SCRIPT = r'''
import io, json, sys

path, start, maximum, column, budget, byte_limit = sys.argv[1:]
start, maximum, column, budget, byte_limit = map(int, (start, maximum, column, budget, byte_limit))
content, end, next_line, next_column = '', None, None, None
number = start
with open(path, 'rb') as original:
    raw = original.read(byte_limit + 1)
if len(raw) > byte_limit:
    raise ValueError('read_file exceeds the 2 MiB file limit; use a targeted command to inspect this file')
original_text = raw.decode('utf-8', errors='replace')
# Recognize complete literal credentials/PEM before any page boundary removes
# their identifying label. Equal-length masking keeps physical cursors stable.
safe_text = redact_file_layout(original_text, path)
with io.StringIO(safe_text, newline='') as source:
    for _ in range(start - 1):
        if not source.readline():
            break
    for _ in range(maximum):
        line = source.readline()
        if not line:
            break
        if column > len(line) + 1:
            raise ValueError('start_column exceeds the line length')
        text = line[column - 1:]
        if len(json.dumps(content + text, ensure_ascii=False)) > budget:
            if not content:
                low, high = 0, len(text)
                while low < high:
                    mid = (low + high + 1) // 2
                    if len(json.dumps(text[:mid], ensure_ascii=False)) <= budget:
                        low = mid
                    else:
                        high = mid - 1
                content = text[:low]
                end = number
                column += low
            next_line, next_column = number, column
            break
        content += text
        end = number
        number, column = number + 1, 1
    else:
        if source.read(1):
            next_line, next_column = number, 1
print(json.dumps(dict(path=path, content=content, redacted=safe_text != original_text, truncated=next_line is not None,
    start_line=start, start_column=int(sys.argv[4]), end_line=end,
    next_line=next_line, next_column=next_column), ensure_ascii=False))
'''


def read_file(arguments, sandbox):
    path = project_path(arguments['path'])
    start = arguments.get('start_line', 1)
    maximum = arguments.get('max_lines', DEFAULT_MAX_LINES)
    column = arguments.get('start_column', 1)
    if any(type(value) is not int or value < 1 for value in (start, maximum, column)):
        raise ValueError('read_file pagination values must be positive integers')
    if maximum > MAX_PAGE_LINES:
        raise ValueError(f'read_file max_lines cannot exceed {MAX_PAGE_LINES}')
    result = sandbox._execute(['python3', '-c', _PRIVACY_SOURCE + '\n' + _PAGE_SCRIPT, path, str(start),
                               str(maximum), str(column), str(CONTENT_BUDGET), str(MAX_SOURCE_BYTES)])
    if result.exit_code != 0:
        return {'error': 'read_file failed', 'exit_code': result.exit_code,
                'stderr': result.stderr.decode('utf-8', errors='replace')[-1500:]}
    return json.loads(result.stdout)
