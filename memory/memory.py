"""LangGraph persistence and small, project-scoped memory operations.

Checkpoints retain one conversation; store items survive across conversations.
Recall uses keyword matching without requiring an embedding service.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore


DEFAULT_DB_PATH = Path.home() / ".local/share/cos30018/memory.sqlite"
MAX_MEMORY_CHARS = 2000
MAX_SEARCH_RESULTS = 20
_PAGE_SIZE = 100
_SENSITIVE_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|password|secret)\b\s*[:=]\s*"
        r"[^\s,;]{8,}",
        re.IGNORECASE,
    ),
)


class SensitiveMemoryError(ValueError):
    """Content appears to contain a credential or private key."""


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryContext:
    """Trusted scope supplied by the application, never by model arguments."""

    project_id: str
    user_id: str = "default"

    def __post_init__(self) -> None:
        _nonempty(self.project_id, "project_id")
        _nonempty(self.user_id, "user_id")

    @property
    def namespace(self) -> tuple[str, ...]:
        return ("projects", _digest(self.project_id), "users", _digest(self.user_id), "memories")


def thread_config(context: MemoryContext, thread_id: str) -> dict:
    """Keep equal conversation names isolated between projects and users."""
    _nonempty(thread_id, "thread_id")
    identity = json.dumps([context.project_id, context.user_id, thread_id])
    return {"configurable": {"thread_id": _digest(identity)}}


@contextmanager
def open_memory(db_path: str | Path | None = None) -> Iterator[tuple[SqliteSaver, SqliteStore]]:
    """Open native checkpoint/store connections for the lifetime of a graph."""
    configured_path = db_path if db_path is not None else os.getenv("MEMORY_DB_PATH")
    path = Path(configured_path or DEFAULT_DB_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        with SqliteStore.from_conn_string(str(path)) as store:
            checkpointer.setup()
            store.setup()
            yield checkpointer, store


def save_memory(store: BaseStore, context: MemoryContext, content: str) -> dict[str, str]:
    """Save a durable fact; the same content in the same scope reuses its ID."""
    content = _nonempty(content, "content").strip()
    if len(content) > MAX_MEMORY_CHARS:
        raise ValueError(f"Memory content cannot exceed {MAX_MEMORY_CHARS} characters")
    if any(pattern.search(content) for pattern in _SENSITIVE_PATTERNS):
        raise SensitiveMemoryError("Memory content appears to contain a credential and was not stored")
    memory_id = _digest(content)
    store.put(context.namespace, memory_id, {"content": content})
    return {"id": memory_id, "content": content}


def _records(store: BaseStore, context: MemoryContext) -> Iterator[dict[str, str]]:
    offset = 0
    while True:
        page = store.search(context.namespace, limit=_PAGE_SIZE, offset=offset)
        for item in page:
            # Native search matches namespace prefixes; restrict to this exact scope.
            if item.namespace == context.namespace and isinstance(item.value.get("content"), str):
                yield {"id": item.key, "content": item.value["content"]}
        if len(page) < _PAGE_SIZE:
            return
        offset += len(page)


def search_memories(
    store: BaseStore, context: MemoryContext, query: str, limit: int = 5
) -> list[dict[str, str]]:
    """Recall memories containing any query word, ignoring case.

    This is lexical retrieval. Native store queries require embeddings for semantic
    search, so fetch paginated records and match keywords explicitly instead.
    """
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    words = set(re.findall(r"\w+", query.casefold()))
    if not words or limit <= 0:
        return []
    results = []
    for record in _records(store, context):
        if words.intersection(re.findall(r"\w+", record["content"].casefold())):
            results.append(record)
            if len(results) == min(limit, MAX_SEARCH_RESULTS):
                break
    return results


def list_memories(store: BaseStore, context: MemoryContext) -> list[dict[str, str]]:
    """List every saved memory in the current project/user scope."""
    return list(_records(store, context))


def forget_memory(store: BaseStore, context: MemoryContext, memory_id: str) -> bool:
    """Delete a memory only from the current project/user scope."""
    _nonempty(memory_id, "memory_id")
    if store.get(context.namespace, memory_id) is None:
        return False
    store.delete(context.namespace, memory_id)
    return True
