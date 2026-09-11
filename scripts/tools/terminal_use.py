"""Execute commands only in the session's Linux sandbox."""
import shlex

WORKSPACE = '/home/user/project'
OUTPUT_LIMIT = 16000


def execute_command(arguments, sandbox):
    script = 'cd ' + WORKSPACE + ' && ' + arguments['command']
    result = sandbox.run_command('timeout --kill-after=5s 30s sh -c ' + shlex.quote(script))
    return {'stdout': result.stdout[:OUTPUT_LIMIT], 'stderr': result.stderr[:OUTPUT_LIMIT],
            'exit_code': result.exit_code, 'timed_out': result.exit_code in (124, 137),
            'truncated': len(result.stdout) > OUTPUT_LIMIT or len(result.stderr) > OUTPUT_LIMIT}
