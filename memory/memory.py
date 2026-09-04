"""Small persistent-memory store used by the first memory experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sqlite3


DEFAULT_DB_PATH = Path(__file__).with_name("memories.db")


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    content: str
    memory_type: str
    created_at: str


class MemoryStore:
    """Persist and retrieve explicitly saved project memories using SQLite FTS5."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured_path = db_path or os.getenv("MEMORY_DB_PATH") or DEFAULT_DB_PATH
        self.db_path = Path(configured_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'decision',
                    created_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS memories_after_insert
                AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_after_delete
                AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_after_update
                AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
                """
            )

    def add(self, content: str, memory_type: str = "decision") -> MemoryRecord:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        if memory_type not in {"decision", "requirement"}:
            raise ValueError("Memory type must be 'decision' or 'requirement'")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memories(content, memory_type, created_at) VALUES (?, ?, ?)",
                (content, memory_type, created_at),
            )
            memory_id = int(cursor.lastrowid)
        return MemoryRecord(memory_id, content, memory_type, created_at)

    def search(self, text: str, limit: int = 3) -> list[MemoryRecord]:
        query = self._fts_query(text)
        if not query or limit <= 0:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.content, m.memory_type, m.created_at
                FROM memories_fts
                JOIN memories AS m ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY bm25(memories_fts), m.created_at DESC
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_all(self) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, content, memory_type, created_at
                FROM memories
                ORDER BY id
                """
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete(self, memory_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
        return cursor.rowcount > 0

    @staticmethod
    def _fts_query(text: str) -> str:
        # Quoting tokens makes punctuation in user prompts safe for MATCH syntax.
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        unique_tokens = list(dict.fromkeys(tokens))[:30]
        return " OR ".join(f'"{token}"' for token in unique_tokens)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            created_at=row["created_at"],
        )


def get_relevant_memories(
    text: str,
    memories: list[str] | None = None,
    *,
    store: MemoryStore | None = None,
    limit: int = 3,
) -> list[str]:
    """Return prompt-ready memory strings.

    ``memories`` is retained as a small compatibility path for older callers.
    New code should pass a ``MemoryStore`` or use the default persistent store.
    """
    if memories is not None:
        query_words = set(re.findall(r"\w+", text.casefold()))
        ranked = sorted(
            memories,
            key=lambda item: len(
                query_words.intersection(re.findall(r"\w+", item.casefold()))
            ),
            reverse=True,
        )
        return [item for item in ranked if query_words.intersection(
            re.findall(r"\w+", item.casefold())
        )][:limit]

    selected_store = store or MemoryStore()
    return [
        f"[{record.memory_type} #{record.id}] {record.content}"
        for record in selected_store.search(text, limit=limit)
    ]
