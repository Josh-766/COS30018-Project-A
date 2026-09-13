"""Persistent, scoped memory for the coding-agent harness.

The store deliberately uses SQLite and FTS5 only.  This keeps the baseline
local and reproducible while leaving semantic/embedding retrieval as a later
extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping


DEFAULT_DB_PATH = Path(__file__).with_name("memories.db")

MEMORY_TYPES = frozenset(
    {
        "requirement",
        "preference",
        "project_fact",
        "decision",
        "constraint",
        "tool_observation",
        "error",
        "bug_fix",
        "review_feedback",
        "episode",
        "lesson",
        "session_summary",
        "procedure",
    }
)
MEMORY_STATUSES = frozenset({"active", "disputed", "superseded", "expired"})

_UNSET = object()
_SENSITIVE_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|password|secret)\b\s*[:=]\s*"
        r"[^\s,;]{8,}",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class MemoryRecord:
    """One durable memory plus its provenance and access scope."""

    id: int
    content: str
    memory_type: str
    created_at: str
    project_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    summary: str | None = None
    source_type: str = "user"
    source_reference: str | None = None
    updated_at: str | None = None
    valid_until: str | None = None
    importance: float = 0.5
    reliability: float = 0.5
    status: str = "active"
    supersedes_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryHit:
    """A ranked and explainable retrieval result."""

    memory_id: int
    content: str
    memory_type: str
    relevance_score: float
    source: str
    reliability: float
    created_at: str
    reason_retrieved: str
    record: MemoryRecord


@dataclass
class TaskState:
    """Transient blackboard state; intentionally separate from durable memory."""

    task_id: str
    project_id: str
    session_id: str
    user_goal: str
    constraints: list[str] = field(default_factory=list)
    current_plan: list[str] = field(default_factory=list)
    active_step: str | None = None
    assigned_agent: str | None = None
    latest_observation: str | None = None
    artifacts: list[str] = field(default_factory=list)
    retry_count: int = 0
    status: str = "pending"
    error: str | None = None


class SensitiveMemoryError(ValueError):
    """Raised when content appears to contain a credential or private key."""


class MemoryStore:
    """Persist and retrieve explicitly saved memories using SQLite FTS5."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured_path = db_path or os.getenv("MEMORY_DB_PATH") or DEFAULT_DB_PATH
        self.db_path = Path(configured_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            fts_already_existed = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memories_fts'"
            ).fetchone() is not None
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'decision',
                    created_at TEXT NOT NULL,
                    project_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    task_id TEXT,
                    agent_id TEXT,
                    summary TEXT,
                    source_type TEXT NOT NULL DEFAULT 'user',
                    source_reference TEXT,
                    updated_at TEXT,
                    valid_until TEXT,
                    importance REAL NOT NULL DEFAULT 0.5,
                    reliability REAL NOT NULL DEFAULT 0.5,
                    status TEXT NOT NULL DEFAULT 'active',
                    supersedes_id INTEGER REFERENCES memories(id) ON DELETE SET NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self._migrate_legacy_schema(connection)
            # Recreate prototype-era triggers so metadata-only updates do not
            # needlessly delete and reinsert FTS entries.
            connection.executescript(
                """
                DROP TRIGGER IF EXISTS memories_after_insert;
                DROP TRIGGER IF EXISTS memories_after_delete;
                DROP TRIGGER IF EXISTS memories_after_update;
                """
            )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS memories_scope_index
                ON memories(project_id, user_id, session_id, task_id, agent_id);
                CREATE INDEX IF NOT EXISTS memories_status_index
                ON memories(status, valid_until);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='id',
                    tokenize='unicode61'
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
                AFTER UPDATE OF content ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
                """
            )
            # Creating triggers does not index rows that predate the FTS table.
            if not fts_already_existed:
                connection.execute(
                    "INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')"
                )

    @staticmethod
    def _migrate_legacy_schema(connection: sqlite3.Connection) -> None:
        """Upgrade databases created by the original four-column prototype."""
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(memories)")
        }
        additions = {
            "project_id": "TEXT",
            "user_id": "TEXT",
            "session_id": "TEXT",
            "task_id": "TEXT",
            "agent_id": "TEXT",
            "summary": "TEXT",
            "source_type": "TEXT NOT NULL DEFAULT 'user'",
            "source_reference": "TEXT",
            "updated_at": "TEXT",
            "valid_until": "TEXT",
            "importance": "REAL NOT NULL DEFAULT 0.5",
            "reliability": "REAL NOT NULL DEFAULT 0.5",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "supersedes_id": "INTEGER REFERENCES memories(id)",
            "metadata": "TEXT NOT NULL DEFAULT '{}'",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE memories ADD COLUMN {name} {definition}")
        connection.execute(
            "UPDATE memories SET updated_at = created_at WHERE updated_at IS NULL"
        )

    def add(
        self,
        content: str,
        memory_type: str = "decision",
        *,
        project_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        summary: str | None = None,
        source_type: str = "user",
        source_reference: str | None = None,
        valid_until: str | datetime | None = None,
        importance: float = 0.5,
        reliability: float = 0.5,
        status: str = "active",
        supersedes_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryRecord:
        """Store a memory, returning an existing exact duplicate if present."""
        content = self._validate_content(content)
        self._validate_choice("memory type", memory_type, MEMORY_TYPES)
        self._validate_choice("status", status, MEMORY_STATUSES)
        importance = self._validate_score("importance", importance)
        reliability = self._validate_score("reliability", reliability)
        valid_until_text = self._normalise_datetime(valid_until)
        metadata_text = self._serialise_metadata(metadata)
        source_type = source_type.strip()
        if not source_type:
            raise ValueError("Source type cannot be empty")

        scope_values = (project_id, user_id, session_id, task_id, agent_id)
        now = self._utc_now()
        with self._connect() as connection:
            duplicate = connection.execute(
                """
                SELECT * FROM memories
                WHERE content = ? AND memory_type = ? AND status = 'active'
                  AND project_id IS ? AND user_id IS ? AND session_id IS ?
                  AND task_id IS ? AND agent_id IS ?
                ORDER BY id DESC LIMIT 1
                """,
                (content, memory_type, *scope_values),
            ).fetchone()
            if duplicate is not None:
                return self._row_to_record(duplicate)

            cursor = connection.execute(
                """
                INSERT INTO memories(
                    content, memory_type, created_at, project_id, user_id,
                    session_id, task_id, agent_id, summary, source_type,
                    source_reference, updated_at, valid_until, importance,
                    reliability, status, supersedes_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content,
                    memory_type,
                    now,
                    *scope_values,
                    summary.strip() if summary else None,
                    source_type,
                    source_reference,
                    now,
                    valid_until_text,
                    importance,
                    reliability,
                    status,
                    supersedes_id,
                    metadata_text,
                ),
            )
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        assert row is not None
        return self._row_to_record(row)

    def get(self, memory_id: int) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def retrieve(
        self,
        text: str,
        limit: int = 3,
        *,
        project_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        include_global: bool = True,
        max_tokens: int | None = None,
    ) -> list[MemoryHit]:
        """Return active, in-scope memories ranked by several local signals."""
        query = self._fts_query(text)
        if not query or limit <= 0 or (max_tokens is not None and max_tokens <= 0):
            return []

        clauses = [
            "memories_fts MATCH ?",
            "m.status = 'active'",
            "(m.valid_until IS NULL OR m.valid_until > ?)",
        ]
        parameters: list[Any] = [query, self._utc_now()]
        for column, value in (
            ("project_id", project_id),
            ("user_id", user_id),
            ("session_id", session_id),
            ("task_id", task_id),
            ("agent_id", agent_id),
        ):
            if value is not None:
                if include_global:
                    clauses.append(f"(m.{column} IS NULL OR m.{column} = ?)")
                else:
                    clauses.append(f"m.{column} = ?")
                parameters.append(value)

        # Pull extra candidates so de-duplication and metadata ranking still have room.
        candidate_limit = max(limit * 8, 24)
        parameters.append(candidate_limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT m.*, bm25(memories_fts) AS keyword_rank
                FROM memories_fts
                JOIN memories AS m ON m.id = memories_fts.rowid
                WHERE {' AND '.join(clauses)}
                ORDER BY keyword_rank, m.created_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        now = datetime.now(timezone.utc)
        ranked: list[tuple[float, MemoryRecord, str]] = []
        for keyword_position, row in enumerate(rows, start=1):
            record = self._row_to_record(row)
            age_days = max(
                0.0, (now - self._parse_datetime(record.updated_at or record.created_at)).total_seconds() / 86400
            )
            recency = math.exp(-age_days / 180.0)
            keyword_score = 1.0 / keyword_position
            score = (
                0.65 * keyword_score
                + 0.15 * record.importance
                + 0.15 * record.reliability
                + 0.05 * recency
            )
            reason = (
                "FTS5 keyword match; ranked using keyword relevance, recency, "
                "importance, and reliability"
            )
            ranked.append((score, record, reason))

        ranked.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        hits: list[MemoryHit] = []
        seen: set[str] = set()
        used_tokens = 0
        for score, record, reason in ranked:
            fingerprint = self._normalise_for_deduplication(record.content)
            if fingerprint in seen:
                continue
            token_cost = self._estimate_tokens(record.content)
            if max_tokens is not None and used_tokens + token_cost > max_tokens:
                continue
            seen.add(fingerprint)
            used_tokens += token_cost
            hits.append(
                MemoryHit(
                    memory_id=record.id,
                    content=record.content,
                    memory_type=record.memory_type,
                    relevance_score=round(score, 6),
                    source=self._source_label(record),
                    reliability=record.reliability,
                    created_at=record.created_at,
                    reason_retrieved=reason,
                    record=record,
                )
            )
            if len(hits) == limit:
                break
        return hits

    def search(self, text: str, limit: int = 3, **scope: Any) -> list[MemoryRecord]:
        """Compatibility wrapper returning records instead of ``MemoryHit`` objects."""
        return [hit.record for hit in self.retrieve(text, limit=limit, **scope)]

    def list_all(
        self,
        *,
        status: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        include_global: bool = True,
    ) -> list[MemoryRecord]:
        if status is not None:
            self._validate_choice("status", status, MEMORY_STATUSES)
        clauses: list[str] = []
        parameters: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        for column, value in (
            ("project_id", project_id),
            ("user_id", user_id),
            ("session_id", session_id),
            ("task_id", task_id),
            ("agent_id", agent_id),
        ):
            if value is not None:
                if include_global:
                    clauses.append(f"({column} IS NULL OR {column} = ?)")
                else:
                    clauses.append(f"{column} = ?")
                parameters.append(value)
        query = "SELECT * FROM memories"
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update(
        self,
        memory_id: int,
        *,
        content: str | None = None,
        summary: str | None | object = _UNSET,
        valid_until: str | datetime | None | object = _UNSET,
        importance: float | None = None,
        reliability: float | None = None,
        status: str | None = None,
        metadata: Mapping[str, Any] | None | object = _UNSET,
    ) -> MemoryRecord | None:
        """Update mutable memory fields and return the new record."""
        assignments = ["updated_at = ?"]
        values: list[Any] = [self._utc_now()]
        if content is not None:
            assignments.append("content = ?")
            values.append(self._validate_content(content))
        if summary is not _UNSET:
            assignments.append("summary = ?")
            values.append(summary.strip() if isinstance(summary, str) else None)
        if valid_until is not _UNSET:
            assignments.append("valid_until = ?")
            values.append(self._normalise_datetime(valid_until))
        if importance is not None:
            assignments.append("importance = ?")
            values.append(self._validate_score("importance", importance))
        if reliability is not None:
            assignments.append("reliability = ?")
            values.append(self._validate_score("reliability", reliability))
        if status is not None:
            self._validate_choice("status", status, MEMORY_STATUSES)
            assignments.append("status = ?")
            values.append(status)
        if metadata is not _UNSET:
            assignments.append("metadata = ?")
            values.append(self._serialise_metadata(metadata))
        values.append(memory_id)

        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE memories SET {', '.join(assignments)} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        assert row is not None
        return self._row_to_record(row)

    def supersede(self, memory_id: int, content: str, **changes: Any) -> MemoryRecord:
        """Replace an active memory while retaining its audit history."""
        previous = self.get(memory_id)
        if previous is None:
            raise KeyError(f"Memory #{memory_id} was not found")
        if previous.status != "active":
            raise ValueError("Only an active memory can be superseded")
        if content.strip() == previous.content:
            raise ValueError("Replacement content must differ from the old memory")

        inherited = {
            "project_id": previous.project_id,
            "user_id": previous.user_id,
            "session_id": previous.session_id,
            "task_id": previous.task_id,
            "agent_id": previous.agent_id,
            "summary": previous.summary,
            "source_type": previous.source_type,
            "source_reference": previous.source_reference,
            "importance": previous.importance,
            "reliability": previous.reliability,
            "metadata": previous.metadata,
        }
        inherited.update(changes)
        inherited["supersedes_id"] = previous.id
        replacement = self.add(content, previous.memory_type, **inherited)
        self.update(previous.id, status="superseded")
        return replacement

    def expire(self, memory_id: int) -> bool:
        return self.update(memory_id, status="expired") is not None

    def delete(self, memory_id: int) -> bool:
        with self._connect() as connection:
            # Legacy databases used a restrictive self-reference. Clear audit
            # links first so an explicit privacy deletion always succeeds.
            connection.execute(
                "UPDATE memories SET supersedes_id = NULL WHERE supersedes_id = ?",
                (memory_id,),
            )
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
        return cursor.rowcount > 0

    @staticmethod
    def _validate_content(content: str) -> str:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        if any(pattern.search(content) for pattern in _SENSITIVE_PATTERNS):
            raise SensitiveMemoryError(
                "Memory content appears to contain a credential and was not stored"
            )
        return content

    @staticmethod
    def _validate_choice(label: str, value: str, choices: frozenset[str]) -> None:
        if value not in choices:
            raise ValueError(f"Invalid {label} {value!r}; expected one of {sorted(choices)}")

    @staticmethod
    def _validate_score(label: str, value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label.capitalize()} must be between 0 and 1")
        return value

    @classmethod
    def _normalise_datetime(cls, value: str | datetime | None | object) -> str | None:
        if value is None or value is _UNSET:
            return None
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("Memory datetimes must include a timezone")
        return parsed.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _serialise_metadata(metadata: Mapping[str, Any] | None | object) -> str:
        try:
            return json.dumps(dict(metadata or {}), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("Memory metadata must be JSON serialisable") from error

    @staticmethod
    def _fts_query(text: str) -> str:
        # Quoting tokens makes punctuation in user prompts safe for MATCH syntax.
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        unique_tokens = list(dict.fromkeys(tokens))[:30]
        return " OR ".join(f'"{token}"' for token in unique_tokens)

    @staticmethod
    def _normalise_for_deduplication(content: str) -> str:
        return " ".join(re.findall(r"\w+", content.casefold(), flags=re.UNICODE))

    @staticmethod
    def _estimate_tokens(content: str) -> int:
        # A conservative dependency-free approximation for prompt budgeting.
        return max(1, math.ceil(len(content) / 4))

    @staticmethod
    def _source_label(record: MemoryRecord) -> str:
        if record.source_reference:
            return f"{record.source_type}:{record.source_reference}"
        return record.source_type

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            created_at=row["created_at"],
            project_id=row["project_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            task_id=row["task_id"],
            agent_id=row["agent_id"],
            summary=row["summary"],
            source_type=row["source_type"],
            source_reference=row["source_reference"],
            updated_at=row["updated_at"],
            valid_until=row["valid_until"],
            importance=float(row["importance"]),
            reliability=float(row["reliability"]),
            status=row["status"],
            supersedes_id=row["supersedes_id"],
            metadata=metadata,
        )


def get_relevant_memories(
    text: str,
    memories: list[str] | None = None,
    *,
    store: MemoryStore | None = None,
    limit: int = 3,
    max_tokens: int | None = None,
    **scope: Any,
) -> list[str]:
    """Return safely delimited, prompt-ready memory strings.

    ``memories`` remains as a compatibility path for older callers. New code
    should pass a ``MemoryStore`` or use the default persistent store.
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
        selected = [
            item
            for item in ranked
            if query_words.intersection(re.findall(r"\w+", item.casefold()))
        ]
        return selected[:limit]

    selected_store = store or MemoryStore()
    return [
        (
            f"[{hit.memory_type} #{hit.memory_id}; source={hit.source}; "
            f"reliability={hit.reliability:.2f}] {hit.content}"
        )
        for hit in selected_store.retrieve(
            text, limit=limit, max_tokens=max_tokens, **scope
        )
    ]
