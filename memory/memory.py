"""Persistent, scoped memory for the coding-agent harness.

SQLite and FTS5 provide an offline baseline. Optional embedding providers
add bounded hybrid retrieval without changing persistence or scope isolation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import heapq
import logging
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator, Mapping

from .embeddings import EmbeddingProvider, VectorCache, normalize_vector, validate_vectors
from .context import estimate_tokens
from .privacy import contains_sensitive_value, redact_sensitive_text


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
_SCOPE_COLUMNS = ("project_id", "user_id", "session_id", "task_id", "agent_id")
_LOGGER = logging.getLogger(__name__)


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
    pending_failures: dict[str, dict[str, Any]] = field(default_factory=dict)


class SensitiveMemoryError(ValueError):
    """Raised when content appears to contain a credential or private key."""


class MemoryStore:
    """Persist and retrieve explicitly saved memories using SQLite FTS5."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        semantic_candidate_limit: int = 256,
    ) -> None:
        configured_path = db_path or os.getenv("MEMORY_DB_PATH") or DEFAULT_DB_PATH
        self.db_path = Path(configured_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider
        if semantic_candidate_limit <= 0:
            raise ValueError("semantic_candidate_limit must be positive")
        # Backward-compatible option: bound ingestion, never cached search coverage.
        self.semantic_candidate_limit = semantic_candidate_limit
        self._query_vectors = VectorCache(max_entries=32, max_coordinates=65_536)
        self._document_vectors = VectorCache(max_entries=512, max_coordinates=1_000_000)
        self._create_schema()

    @contextmanager
    def _connect(self, *, write: bool = False, initialise: bool = False) -> Iterator[sqlite3.Connection]:
        """One short-lived connection; commit/rollback and close on every path."""
        connection = sqlite3.connect(self.db_path, timeout=15.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 15000")
            if initialise:
                connection.execute("PRAGMA journal_mode = WAL")
            if write:
                # Serialize read-modify-write operations, including deduplication.
                connection.execute("BEGIN IMMEDIATE")
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _execute_schema_script(connection: sqlite3.Connection, script: str) -> None:
        # executescript() commits implicitly, which would break atomic migrations.
        statement = ""
        for line in script.splitlines(keepends=True):
            statement += line
            if sqlite3.complete_statement(statement):
                connection.execute(statement)
                statement = ""

    def _create_schema(self) -> None:
        with self._connect(write=True, initialise=True) as connection:
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
            self._execute_schema_script(connection,
                """
                DROP TRIGGER IF EXISTS memories_after_insert;
                DROP TRIGGER IF EXISTS memories_after_delete;
                DROP TRIGGER IF EXISTS memories_after_update;
                """
            )
            self._execute_schema_script(connection,
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    model_fingerprint TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY (memory_id, model_fingerprint)
                );

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
        """Save a validated memory atomically, reusing only unexpired duplicates."""
        payload = self._prepare_payload({
            "content": content, "memory_type": memory_type,
            "project_id": project_id, "user_id": user_id,
            "session_id": session_id, "task_id": task_id, "agent_id": agent_id,
            "summary": summary, "source_type": source_type,
            "source_reference": source_reference, "valid_until": valid_until,
            "importance": importance, "reliability": reliability,
            "status": status, "supersedes_id": supersedes_id, "metadata": metadata,
        })
        with self._connect(write=True) as connection:
            return self._insert_memory(connection, payload)

    @classmethod
    def _prepare_payload(cls, fields: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **dict.fromkeys(_SCOPE_COLUMNS), "summary": None,
            "source_type": "user", "source_reference": None, "valid_until": None,
            "importance": 0.5, "reliability": 0.5, "status": "active",
            "supersedes_id": None, "metadata": None,
            "content": "", "memory_type": "decision",
        }
        unknown = fields.keys() - payload.keys()
        if unknown:
            raise TypeError(f"Unsupported memory fields: {', '.join(sorted(unknown))}")
        payload.update(fields)
        payload["content"] = cls._validate_content(payload["content"])
        cls._validate_choice("memory type", payload["memory_type"], MEMORY_TYPES)
        cls._validate_choice("status", payload["status"], MEMORY_STATUSES)
        for name in ("importance", "reliability"):
            payload[name] = cls._validate_score(name, payload[name])
        payload["valid_until"] = cls._normalise_datetime(payload["valid_until"])
        for name in (*_SCOPE_COLUMNS, "summary", "source_type", "source_reference"):
            value = payload[name]
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string or None")
                cls._validate_sensitive_value(value)
                payload[name] = value.strip() or None
        if not payload["source_type"]:
            raise ValueError("Source type cannot be empty")
        payload["metadata"] = cls._serialise_metadata(payload["metadata"])
        return payload

    def _insert_memory(self, connection: sqlite3.Connection, payload: dict[str, Any]) -> MemoryRecord:
        now = self._utc_now()
        previous_id = payload["supersedes_id"]
        if previous_id is not None:
            previous = connection.execute("SELECT * FROM memories WHERE id = ?", (previous_id,)).fetchone()
            if previous is None:
                raise KeyError(f"Memory #{previous_id} was not found")
            if previous["status"] != "active" or payload["status"] != "active":
                raise ValueError("Supersession requires active old and replacement memories")
            if any(previous[column] != payload[column] for column in _SCOPE_COLUMNS):
                raise ValueError("A replacement cannot change its memory scope")
            if previous["memory_type"] != payload["memory_type"]:
                raise ValueError("A replacement cannot change its memory type")
            if previous["content"] == payload["content"]:
                raise ValueError("Replacement content must differ from the old memory")
        duplicate = connection.execute(
            """SELECT * FROM memories
               WHERE content = ? AND memory_type = ? AND status = 'active'
                 AND (valid_until IS NULL OR valid_until > ?)
                 AND project_id IS ? AND user_id IS ? AND session_id IS ?
                 AND task_id IS ? AND agent_id IS ? ORDER BY id DESC LIMIT 1""",
            (payload["content"], payload["memory_type"], now,
             *(payload[name] for name in _SCOPE_COLUMNS)),
        ).fetchone()
        if duplicate is not None and payload["status"] == "active":
            if previous_id is not None:
                raise ValueError("Replacement duplicates an existing active memory")
            return self._refresh_duplicate(connection, duplicate, payload, now)
        payload = {**payload, "created_at": now, "updated_at": now}
        columns = ", ".join(payload)
        placeholders = ", ".join("?" for _ in payload)
        cursor = connection.execute(
            f"INSERT INTO memories ({columns}) VALUES ({placeholders})", tuple(payload.values())
        )
        if previous_id is not None:
            connection.execute(
                "UPDATE memories SET status = 'superseded', updated_at = ? WHERE id = ?",
                (now, previous_id),
            )
            connection.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (previous_id,))
        row = connection.execute("SELECT * FROM memories WHERE id = ?", (cursor.lastrowid,)).fetchone()
        assert row is not None
        return self._row_to_record(row)

    def _refresh_duplicate(self, connection: sqlite3.Connection, previous: sqlite3.Row,
                           payload: dict[str, Any], now: str) -> MemoryRecord:
        """Reconfirmation renews evidence without multiplying identical facts.

        Weaker sources cannot rewrite user provenance or extend its lifetime.
        Source labels are supplied by trusted application code, not agent tools.
        """
        authority = {'user': 3, 'document': 2, 'tool': 1}
        old_authority = authority.get(previous['source_type'], 0)
        new_authority = authority.get(payload['source_type'], 0)
        expiry = payload['valid_until']
        if (new_authority < old_authority or payload['reliability'] < previous['reliability']
                or (expiry is not None and expiry <= now)):
            return self._row_to_record(previous)
        if new_authority == old_authority:
            old_expiry = previous['valid_until']
            expiry = max(old_expiry, expiry) if old_expiry and expiry else None
        metadata = {**self._row_to_record(previous).metadata, **json.loads(payload['metadata'])}
        connection.execute('''UPDATE memories SET updated_at = ?, source_type = ?,
            source_reference = ?, reliability = ?, importance = ?, valid_until = ?,
            summary = ?, metadata = ? WHERE id = ?''',
            (now, payload['source_type'], payload['source_reference'], payload['reliability'],
             max(previous['importance'], payload['importance']), expiry,
             payload['summary'] if payload['summary'] is not None else previous['summary'],
             json.dumps(metadata, sort_keys=True, allow_nan=False), previous['id']))
        return self._row_to_record(connection.execute(
            'SELECT * FROM memories WHERE id = ?', (previous['id'],)).fetchone())

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

        scope_clauses, scope_parameters = self._scope_filter(
            (project_id, user_id, session_id, task_id, agent_id), include_global, alias="m."
        )
        active_clauses = ["m.status = 'active'", "(m.valid_until IS NULL OR m.valid_until > ?)", *scope_clauses]
        active_parameters: list[Any] = [self._utc_now(), *scope_parameters]
        clauses = ["memories_fts MATCH ?", *active_clauses]
        parameters: list[Any] = [query, *active_parameters]

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

        records = {row["id"]: self._row_to_record(row) for row in rows}
        keyword_scores = {row["id"]: 1.0 / position for position, row in enumerate(rows, start=1)}
        semantic_scores: dict[int, float] = {}
        if self.embedding_provider is not None:
            try:
                semantic_records, semantic_scores = self._semantic_candidates(
                    text, active_clauses, active_parameters,
                    keyword_scores=keyword_scores, candidate_limit=candidate_limit,
                )
                records.update(semantic_records)
            except Exception as error:
                # Optional provider failures must never break local memory access.
                # Do not log provider messages: they may contain prompts or secrets.
                _LOGGER.warning("Semantic memory retrieval unavailable (%s); using FTS5", type(error).__name__)
        if records:
            with self._connect() as connection:
                current = connection.execute(
                    f"SELECT m.* FROM memories m WHERE {' AND '.join(active_clauses)} AND m.id IN ({','.join('?' for _ in records)})",
                    [self._utc_now(), *scope_parameters, *records],
                ).fetchall()
            records = {row["id"]: self._row_to_record(row) for row in current if row["content"] == records[row["id"]].content}
        now = datetime.now(timezone.utc)
        ranked: list[tuple[float, MemoryRecord, str]] = []
        for memory_id, record in records.items():
            age_days = max(0.0, (now - self._parse_datetime(record.updated_at or record.created_at)).total_seconds() / 86400)
            recency = math.exp(-age_days / 180.0)
            keyword_score = keyword_scores.get(memory_id, 0.0)
            semantic_score = semantic_scores.get(memory_id, 0.0)
            relevance = (0.55 * keyword_score + 0.45 * semantic_score) if semantic_scores else keyword_score
            score = (0.65 * relevance + 0.15 * record.importance + 0.15 * record.reliability + 0.05 * recency)
            signals = []
            if keyword_score:
                signals.append("FTS5 keyword match")
            if semantic_score:
                signals.append(f"semantic cosine={semantic_score:.3f}")
            reason = "; ".join(signals) + "; ranked using relevance, recency, importance, and reliability"
            ranked.append((score, record, reason))

        ranked.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        hits: list[MemoryHit] = []
        seen: set[str] = set()
        used_tokens = 0
        for score, record, reason in ranked:
            fingerprint = record.memory_type + ":" + self._normalise_for_deduplication(record.content)
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

    @staticmethod
    def _scope_filter(
        values: tuple[str | None, ...], include_global: bool, *, alias: str = ""
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in zip(_SCOPE_COLUMNS, values):
            if value is None:
                clauses.append(f"{alias}{column} IS NULL")
            elif include_global:
                clauses.append(f"({alias}{column} IS NULL OR {alias}{column} = ?)")
                parameters.append(value)
            else:
                clauses.append(f"{alias}{column} = ?")
                parameters.append(value)
        return clauses, parameters

    def _semantic_candidates(
        self, text: str, clauses: list[str], parameters: list[Any], *,
        keyword_scores: Mapping[int, float], candidate_limit: int,
    ) -> tuple[dict[int, MemoryRecord], dict[int, float]]:
        """Search all eligible cached vectors; separately bound new embeddings.

        Database reads/writes finish before provider calls. Valid batches commit
        independently, so a later failure cannot discard already paid-for work.
        Only a bounded heap of matching records is retained during the scan.
        """
        provider = self.embedding_provider
        assert provider is not None
        self._validate_sensitive_value(text)
        fingerprint = provider.fingerprint
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Embedding provider must expose a nonempty fingerprint")
        with self._connect() as connection:
            exists = connection.execute(
                f"SELECT 1 FROM memories m WHERE {' AND '.join(clauses)} LIMIT 1", parameters,
            ).fetchone()
        if exists is None:
            return {}, {}

        # Query failures incur no document work. Repeated model rounds reuse the
        # same vector without persisting the user prompt in the database.
        query_key = (fingerprint, hashlib.sha256(text.encode("utf-8")).hexdigest())
        query_vector = self._query_vectors.get(query_key)
        if query_vector is None:
            query_vector = normalize_vector(validate_vectors(provider.embed([text]), 1)[0])
            self._query_vectors.put(query_key, query_vector)
        dimension = len(query_vector)
        matches: list[tuple[float, int, MemoryRecord, float]] = []
        now = datetime.now(timezone.utc)

        def remember(row: sqlite3.Row, vector: tuple[float, ...]) -> None:
            similarity = max(-1.0, min(1.0, sum(a * b for a, b in zip(query_vector, vector))))
            if similarity < 0.55:
                return
            record = self._row_to_record(row)
            age = max(0.0, (now - self._parse_datetime(record.updated_at or record.created_at)).total_seconds() / 86400)
            relevance = 0.55 * keyword_scores.get(record.id, 0.0) + 0.45 * similarity
            score = (0.65 * relevance + 0.15 * record.importance
                     + 0.15 * record.reliability + 0.05 * math.exp(-age / 180.0))
            item = (score, record.id, record, similarity)
            if len(matches) < candidate_limit:
                heapq.heappush(matches, item)
            elif item[:2] > matches[0][:2]:
                heapq.heapreplace(matches, item)

        # Stream cached rows, rather than loading every JSON vector at once or
        # limiting comparison to the newest N memories. Hashes guard edits and
        # cache corruption; normalized vectors avoid repeated parsing/norm work.
        missing: list[sqlite3.Row] = []
        with self._connect() as connection:
            cached = connection.execute(
                f"""SELECT m.*, e.content_hash AS embedded_hash, e.vector AS embedded_vector
                    FROM memories m JOIN memory_embeddings e
                    ON e.memory_id = m.id AND e.model_fingerprint = ?
                    WHERE {' AND '.join(clauses)}""",
                [fingerprint, *parameters],
            )
            for row in cached:
                content_hash = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
                try:
                    if row["embedded_hash"] != content_hash:
                        raise ValueError("Stale embedding")
                    serialized = row["embedded_vector"]
                    vector_key = (fingerprint, str(row["id"]), content_hash,
                                  hashlib.sha256(serialized.encode("utf-8")).hexdigest())
                    vector = self._document_vectors.get(vector_key)
                    if vector is None:
                        vector = normalize_vector(json.loads(serialized))
                        self._document_vectors.put(vector_key, vector)
                    if len(vector) != dimension:
                        raise ValueError("Embedding dimensions do not match")
                    remember(row, vector)
                except (ValueError, TypeError, AttributeError):
                    if len(missing) < self.semantic_candidate_limit:
                        missing.append(row)
            remaining = self.semantic_candidate_limit - len(missing)
            if remaining:
                missing.extend(connection.execute(
                    f"""SELECT m.* FROM memories m LEFT JOIN memory_embeddings e
                        ON e.memory_id = m.id AND e.model_fingerprint = ?
                        WHERE {' AND '.join(clauses)} AND e.memory_id IS NULL
                        ORDER BY m.importance DESC, m.updated_at DESC, m.id DESC LIMIT ?""",
                    [fingerprint, *parameters, remaining],
                ).fetchall())

        for offset in range(0, len(missing), 32):
            batch = missing[offset:offset + 32]
            # Refresh eligibility/content immediately before sending a batch.
            with self._connect() as connection:
                batch = connection.execute(
                    f"""SELECT m.* FROM memories m WHERE {' AND '.join(clauses)}
                        AND m.id IN ({','.join('?' for _ in batch)})""",
                    [self._utc_now(), *parameters[1:], *(row["id"] for row in batch)],
                ).fetchall()
            if not batch:
                continue
            try:
                for row in batch:
                    self._validate_sensitive_value(row["content"])
                generated = validate_vectors(provider.embed([row["content"] for row in batch]), len(batch))
                if any(len(vector) != dimension for vector in generated):
                    raise ValueError("Embedding dimensions do not match")
                normalized = [normalize_vector(vector) for vector in generated]
            except Exception as error:
                _LOGGER.warning("Semantic memory ingestion unavailable (%s); retaining cached matches", type(error).__name__)
                break
            with self._connect(write=True) as connection:
                for row, vector, unit in zip(batch, generated, normalized):
                    content_hash = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
                    serialized = json.dumps(vector, separators=(",", ":"))
                    # Concurrent changes must not restore deleted/expired data
                    # or associate an old embedding with replacement content.
                    inserted = connection.execute(
                        f"""INSERT OR REPLACE INTO memory_embeddings
                            (memory_id, model_fingerprint, content_hash, vector)
                            SELECT m.id, ?, ?, ? FROM memories m
                            WHERE {' AND '.join(clauses)} AND m.id = ? AND m.content = ?""",
                        [fingerprint, content_hash, serialized, self._utc_now(),
                         *parameters[1:], row["id"], row["content"]],
                    )
                    if inserted.rowcount:
                        self._document_vectors.put(
                            (fingerprint, str(row["id"]), content_hash,
                             hashlib.sha256(serialized.encode("utf-8")).hexdigest()), unit,
                        )
                        remember(row, unit)
        return ({item[1]: item[2] for item in matches}, {item[1]: item[3] for item in matches})

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
        scope_values = (project_id, user_id, session_id, task_id, agent_id)
        # No scope is an explicit administrative inventory. Scoped calls always
        # exclude identities the caller did not supply, just like retrieve().
        if any(value is not None for value in scope_values):
            scope_clauses, scope_parameters = self._scope_filter(scope_values, include_global)
            clauses.extend(scope_clauses)
            parameters.extend(scope_parameters)
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
            if summary is not None and not isinstance(summary, str):
                raise ValueError("summary must be a string or None")
            self._validate_sensitive_value(summary)
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

        with self._connect(write=True) as connection:
            previous = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if previous is None:
                return None
            proposed_content = self._validate_content(content) if content is not None else previous["content"]
            proposed_status = status if status is not None else previous["status"]
            proposed_expiry = self._normalise_datetime(valid_until) if valid_until is not _UNSET else previous["valid_until"]
            if proposed_status == "active" and (proposed_expiry is None or proposed_expiry > self._utc_now()):
                duplicate = connection.execute(
                    """SELECT id FROM memories WHERE id != ? AND content = ?
                       AND memory_type = ? AND status = 'active'
                       AND (valid_until IS NULL OR valid_until > ?)
                       AND project_id IS ? AND user_id IS ? AND session_id IS ?
                       AND task_id IS ? AND agent_id IS ? LIMIT 1""",
                    (memory_id, proposed_content, previous["memory_type"], self._utc_now(),
                     *(previous[name] for name in _SCOPE_COLUMNS)),
                ).fetchone()
                if duplicate is not None:
                    raise ValueError("Update duplicates an existing active memory")
            cursor = connection.execute(
                f"UPDATE memories SET {', '.join(assignments)} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                return None
            if content is not None or status is not None:
                connection.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        assert row is not None
        return self._row_to_record(row)

    def supersede(self, memory_id: int, content: str, **changes: Any) -> MemoryRecord:
        """Atomically replace a memory in its original scope, retaining history."""
        with self._connect(write=True) as connection:
            previous = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if previous is None:
                raise KeyError(f"Memory #{memory_id} was not found")
            inherited = {name: previous[name] for name in (
                *_SCOPE_COLUMNS, "summary", "source_type", "source_reference",
                "importance", "reliability", "valid_until", "memory_type",
            )}
            inherited["metadata"] = json.loads(previous["metadata"] or "{}")
            inherited.update(changes)
            if "supersedes_id" in changes and changes["supersedes_id"] != memory_id:
                raise ValueError("Cannot override the memory being superseded")
            payload = self._prepare_payload({**inherited, "content": content, "supersedes_id": memory_id})
            return self._insert_memory(connection, payload)

    def expire(self, memory_id: int) -> bool:
        return self.update(memory_id, status="expired") is not None

    def delete(self, memory_id: int) -> bool:
        with self._connect(write=True) as connection:
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

    @classmethod
    def _validate_content(cls, content: str) -> str:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Memory content cannot be empty")
        cls._validate_sensitive_value(content)
        return content.strip()

    @classmethod
    def _validate_sensitive_value(cls, value: Any) -> None:
        """Use the same source-aware detector for writes and active context."""
        if contains_sensitive_value(value):
            raise SensitiveMemoryError("Memory payload appears to contain a credential and was not stored")

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

    @classmethod
    def _serialise_metadata(cls, metadata: Mapping[str, Any] | None | object) -> str:
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("Memory metadata must be a JSON object")
        try:
            serialised = json.dumps(dict(metadata or {}), sort_keys=True, allow_nan=False)
            cls._validate_sensitive_value(metadata)
            return serialised
        except SensitiveMemoryError:
            raise
        except (TypeError, ValueError) as error:
            raise ValueError("Memory metadata must be JSON serialisable") from error

    @staticmethod
    def _fts_query(text: str) -> str:
        """Keep informative coding terms across the prompt in a bounded MATCH.

        Error codes, symbols and path components outrank prose. If a bucket is
        oversized, sample across it (including its tail) instead of silently
        discarding diagnostics appended after a long task description.
        """
        stopwords = {"a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "should", "that", "the", "this", "to", "was", "we", "what", "when", "where", "which", "with", "you"}
        candidates: dict[str, tuple[int, int]] = {}
        for match in re.finditer(r"\w+", text, flags=re.UNICODE):
            raw = match.group()
            token = raw.casefold()
            if token in stopwords or len(token) > 128:
                continue
            priority = 0
            if (len(raw) >= 3 and raw.isupper() and any(character.isalpha() for character in raw)):
                priority = 4
            elif (any(character.isdigit() for character in raw) and any(character.isalpha() for character in raw)
                  or "_" in raw or re.search(r"[a-z][A-Z]", raw)
                  or raw.endswith(("Error", "Exception"))):
                priority = 3
            elif (match.start() and text[match.start() - 1] in "/\\.`"
                  or match.end() < len(text) and text[match.end()] in "/\\.`"):
                priority = 2
            previous = candidates.get(token)
            if previous is None or priority > previous[0]:
                candidates[token] = (priority, match.start())
        selected: list[str] = []
        for priority in (4, 3, 2, 0):
            bucket = [token for token, rank in candidates.items() if rank[0] == priority]
            remaining = 48 - len(selected)
            if remaining <= 0:
                break
            if len(bucket) <= remaining:
                selected.extend(bucket)
            elif remaining == 1:
                selected.append(bucket[-1])
            else:
                selected.extend(bucket[round(index * (len(bucket) - 1) / (remaining - 1))] for index in range(remaining))
        # Tokens contain only word characters; quoting also blocks MATCH syntax.
        return " OR ".join(f'"{token}"' for token in selected)

    @staticmethod
    def _normalise_for_deduplication(content: str) -> str:
        # Preserve punctuation and letter case: C++, C#, paths and code can
        # differ meaningfully even when their word tokens look the same.
        return content.strip()

    @staticmethod
    def _estimate_tokens(content: str) -> int:
        return estimate_tokens(content)

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
        if not isinstance(metadata, dict):
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
    """Return legacy display strings, with a budget including their metadata.

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
    else:
        selected_store = store or MemoryStore()
        selected = [
            (f"[{hit.memory_type} #{hit.memory_id}; source={hit.source}; "
             f"reliability={hit.reliability:.2f}] {hit.content}")
            for hit in selected_store.retrieve(text, limit=limit, **scope)
        ]
    bounded: list[str] = []
    for content in selected:
        if len(bounded) >= max(0, limit):
            break
        if max_tokens is None or estimate_tokens('\n'.join([*bounded, content])) <= max_tokens:
            bounded.append(content)
    return bounded
