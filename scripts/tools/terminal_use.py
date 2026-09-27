"""Execute commands only in the session's Linux sandbox."""
import shlex
from types import SimpleNamespace
from memory.privacy import redact_sensitive_text

WORKSPACE = '/home/user/project'
OUTPUT_LIMIT = 16000


TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run a Linux shell command; 30 second timeout.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string"
                }
            },
            "required": [
                "command"
            ],
            "additionalProperties": False
        }
    }
}


def run_command(command, sandbox, timeout=60):
    result = sandbox._execute(['sh', '-c', command], timeout)
    return SimpleNamespace(stdout=result.stdout.decode('utf-8', errors='replace'),
                           stderr=result.stderr.decode('utf-8', errors='replace'),
                           exit_code=result.exit_code)


def execute_command(arguments, sandbox):
    script = 'cd ' + WORKSPACE + ' && ' + arguments['command']
    result = run_command('timeout --kill-after=5s 30s sh -c ' + shlex.quote(script), sandbox)
    # Detect complete literals/PEM blocks before clipping their identifying prefix.
    stdout, stderr = redact_sensitive_text(result.stdout), redact_sensitive_text(result.stderr)
    def bounded(stream):
        if len(stream) <= OUTPUT_LIMIT:
            return stream
        marker = '\n...[output omitted]...\n'
        head = (OUTPUT_LIMIT - len(marker)) // 2
        tail = OUTPUT_LIMIT - len(marker) - head
        return stream[:head] + marker + stream[-tail:]
    return {'stdout': bounded(stdout), 'stderr': bounded(stderr),
            'exit_code': result.exit_code, 'timed_out': result.exit_code in (124, 137),
            'truncated': len(stdout) > OUTPUT_LIMIT or len(stderr) > OUTPUT_LIMIT,
            'redacted': stdout != result.stdout or stderr != result.stderr}
