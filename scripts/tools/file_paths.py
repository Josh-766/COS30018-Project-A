"""Shared path handling for file tools."""
from pathlib import PurePosixPath
from .terminal_use import WORKSPACE


def project_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError("Use a relative project path without '..'.")
    return str(PurePosixPath(WORKSPACE) / path)
