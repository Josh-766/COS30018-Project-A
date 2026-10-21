import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

from .terminal_use import WORKSPACE

EXCLUDED = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.ssh', '.aws',
            '.codex', '.agents', 'sandbox-output', '.pytest_cache'}

# kubectl cp runs tar with an absolute pod path, so tar prints this warning
# for every copy. It arrives on stdout, not stderr.
TAR_NOISE = re.compile(r'^tar: Removing leading .+$')


def _strip_tar_noise(stream):
    if not stream:
        return stream
    is_text = isinstance(stream, str)
    kept = [
        line for line in stream.splitlines(keepends=True)
        if not TAR_NOISE.match(
            line.strip() if is_text else line.decode(errors='replace').strip()
        )
    ]
    return (''.join(kept) if is_text else b''.join(kept))


def _echo_stream(stream, target):
    if not stream:
        return
    if isinstance(stream, str):
        target.write(stream)
    else:
        buffer = getattr(target, 'buffer', None)
        if buffer is None:
            target.write(stream.decode(errors='replace'))
        else:
            buffer.write(stream)
    target.flush()


class Sandbox:
    def __init__(self, pod_name, namespace, profile):
        self.pod_name = pod_name
        self.namespace = namespace
        self.profile = profile


    def save_to_host(self, project: Path) -> None:

        self._kubectl(
            [
                "cp",
                "-c", "sandbox",
                f"{self.pod_name}:/home/user/project/.",
                ".",
            ],
            cwd=project,
            check=True,
            timeout=180,
        )
    def _kubectl(self, arguments, **kwargs):
        command = ['kubectl', '--context', self.profile, '--namespace', self.namespace] + arguments
        # kubectl cp writes tar's "Removing leading..." warnings to stdout.
        # Filter those while leaving other commands and real diagnostics alone.
        if arguments and arguments[0] == 'cp' and not kwargs.get('capture_output'):
            check = kwargs.pop('check', False)
            kwargs.pop('capture_output', None)
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
            completed = subprocess.run(command, check=False, **kwargs)

            completed.stdout = _strip_tar_noise(completed.stdout)
            completed.stderr = _strip_tar_noise(completed.stderr)

            _echo_stream(completed.stdout, sys.stdout)
            _echo_stream(completed.stderr, sys.stderr)

            if check and completed.returncode:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    command,
                    output=completed.stdout,
                    stderr=completed.stderr,
                )
            return completed
        return subprocess.run(command, **kwargs)

    def _execute(self, command, timeout=60, input_data=None):
        result = self._kubectl(
            ['exec', '-i', self.pod_name, '-c', 'sandbox', '--',
             'timeout', '--kill-after=5s', str(timeout) + 's'] + command,
            input=input_data if input_data is not None else b'',
            capture_output=True, timeout=timeout + 10,
        )
        return SimpleNamespace(stdout=result.stdout, stderr=result.stderr,
                               exit_code=result.returncode)

    def terminate(self):
        self._kubectl(['delete', 'pod', self.pod_name, '--ignore-not-found=true',
                       '--wait=false'], check=True, timeout=30)


def project_files(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in EXCLUDED and not d.startswith('.env')
                   and not (Path(directory) / d).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if (path.is_symlink() or name.startswith('.env')
                    or path.suffix.lower() in {'.pem', '.key', '.pfx'}):
                continue
            yield path


def create_session(project):
    profile = os.getenv('MINIKUBE_PROFILE', 'minikube')
    namespace = os.getenv('SANDBOX_NAMESPACE', 'agent-sandbox')
    print(f'Starting Minikube profile: {profile}')
    subprocess.run(['minikube', 'start', '-p', profile], check=True, timeout=300)

    pod_name = 'coding-sandbox-' + uuid.uuid4().hex[:8]
    sandbox = Sandbox(pod_name, namespace, profile)
    sandbox._kubectl(['apply', '-f', '-'], input=json.dumps({
        'apiVersion': 'v1', 'kind': 'Namespace', 'metadata': {'name': namespace},
    }), text=True, check=True, timeout=30)
    pod = Path(__file__).with_name('sandbox.yaml').read_text(encoding='utf-8')
    pod = pod.replace('${POD_NAME}', json.dumps(pod_name))
    pod = pod.replace('${SANDBOX_IMAGE}', json.dumps(os.getenv('SANDBOX_IMAGE', 'python:3.12-slim')))
    try:
        sandbox._kubectl(['create', '-f', '-'], input=pod, text=True, check=True, timeout=30)
        sandbox._kubectl(['wait', '--for=condition=Ready', 'pod/' + pod_name,
                           '--timeout=180s'], check=True, timeout=190)
        with tempfile.TemporaryDirectory() as directory:
            for path in project_files(project):
                destination = Path(directory) / path.relative_to(project)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            sandbox._kubectl(['cp', '--no-preserve', '-c', 'sandbox', '.',
                               pod_name + ':' + WORKSPACE], cwd=directory, check=True, timeout=180)
        print(f'Sandbox ready: {pod_name}')
        return sandbox
    except BaseException:
        sandbox.terminate()
        raise
