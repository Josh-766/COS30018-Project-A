from concurrent.futures import ThreadPoolExecutor
import io
import json
import tempfile
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import unittest

from memory import MemoryStore, SensitiveMemoryError, get_relevant_memories
from memory.embeddings import OpenRouterEmbeddingProvider
from memory.memory import redact_sensitive_text


class MemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "memory.db"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_memory_survives_new_store_instance(self) -> None:
        first_process = MemoryStore(self.db_path)
        saved = first_process.add("All API requests require a 60-second timeout")

        restarted_process = MemoryStore(self.db_path)
        results = restarted_process.search("What timeout should API requests use?")

        self.assertEqual(results, [saved])

    def test_retrieval_returns_only_relevant_memories(self) -> None:
        store = MemoryStore(self.db_path)
        store.add("The project must use Python 3.12", "requirement")
        store.add("The user interface uses a dark colour palette", "decision")

        results = get_relevant_memories("Which Python version?", store=store)

        self.assertEqual(len(results), 1)
        self.assertIn("Python 3.12", results[0])
        self.assertIn("requirement", results[0])

    def test_delete_removes_memory_from_search(self) -> None:
        store = MemoryStore(self.db_path)
        saved = store.add("Use SQLite for persistent memory")

        self.assertTrue(store.delete(saved.id))
        self.assertEqual(store.search("Which SQLite database?"), [])
        self.assertFalse(store.delete(saved.id))

    def test_retrieval_isolates_project_scope_and_includes_global_memory(self) -> None:
        store = MemoryStore(self.db_path)
        global_memory = store.add("All projects use reviewer approval")
        project_a = store.add(
            "Project A uses FastAPI", "project_fact", project_id="project-a"
        )
        store.add("Project B uses Flask", "project_fact", project_id="project-b")

        results = store.search("Which project uses API reviewer?", project_id="project-a")

        self.assertEqual({record.id for record in results}, {global_memory.id, project_a.id})

    def test_expired_and_superseded_memories_are_not_retrieved(self) -> None:
        store = MemoryStore(self.db_path)
        expired = store.add(
            "Deploy with version one",
            valid_until=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        old = store.add("The timeout is thirty seconds")
        replacement = store.supersede(old.id, "The timeout is sixty seconds")

        results = store.search("What timeout version should we use?", limit=10)

        self.assertNotIn(expired.id, {record.id for record in results})
        self.assertNotIn(old.id, {record.id for record in results})
        self.assertIn(replacement.id, {record.id for record in results})
        self.assertEqual(replacement.supersedes_id, old.id)
        self.assertEqual(store.get(old.id).status, "superseded")  # type: ignore[union-attr]

        self.assertTrue(store.delete(old.id))
        self.assertIsNone(store.get(replacement.id).supersedes_id)  # type: ignore[union-attr]

    def test_updating_content_refreshes_the_search_index(self) -> None:
        store = MemoryStore(self.db_path)
        saved = store.add("Use the Falcon framework")

        updated = store.update(saved.id, content="Use the FastAPI framework")

        self.assertIsNotNone(updated)
        self.assertEqual(store.search("Falcon"), [])
        self.assertEqual(store.search("FastAPI"), [updated])

    def test_retrieve_returns_provenance_and_obeys_token_budget(self) -> None:
        store = MemoryStore(self.db_path)
        saved = store.add(
            "Python 3.12",
            "requirement",
            source_type="document",
            source_reference="assignment.pdf#page=2",
            reliability=0.9,
        )

        hits = store.retrieve("Python version", max_tokens=4)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].memory_id, saved.id)
        self.assertEqual(hits[0].source, "document:assignment.pdf#page=2")
        self.assertGreater(hits[0].relevance_score, 0)

    def test_exact_duplicate_in_same_scope_is_not_inserted_twice(self) -> None:
        store = MemoryStore(self.db_path)
        first = store.add("Use SQLite", project_id="project-a")
        duplicate = store.add("  Use SQLite  ", project_id="project-a")
        other_scope = store.add("Use SQLite", project_id="project-b")

        self.assertEqual(first.id, duplicate.id)
        self.assertNotEqual(first.id, other_scope.id)
        self.assertEqual(len(store.list_all()), 2)

    def test_credentials_are_rejected(self) -> None:
        store = MemoryStore(self.db_path)

        with self.assertRaises(SensitiveMemoryError):
            store.add("api_key=sk-example-secret-value")

        self.assertEqual(store.list_all(), [])

    def test_legacy_database_is_migrated_and_indexed(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'decision',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO memories(content, memory_type, created_at) VALUES (?, ?, ?)",
                ("Legacy memory uses SQLite", "decision", datetime.now(timezone.utc).isoformat()),
            )

        store = MemoryStore(self.db_path)

        results = store.search("SQLite")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_type, "user")
        self.assertEqual(results[0].status, "active")


    def test_missing_identity_never_retrieves_private_memories(self) -> None:
        store = MemoryStore(self.db_path)
        public = store.add("Shared Python rule", project_id="a")
        global_record = store.add("Global Python rule")
        for identity in ("user_id", "session_id", "task_id", "agent_id"):
            store.add(f"Private Python rule {identity}", project_id="a", **{identity: "private"})
            store.add(f"Global private Python rule {identity}", **{identity: "private"})
        self.assertEqual({item.id for item in store.search("Python", project_id="a", limit=20)}, {public.id, global_record.id})
        self.assertEqual({item.id for item in store.list_all(project_id="a")}, {public.id, global_record.id})
        self.assertEqual(store.search("Python", limit=20), [global_record])
        self.assertEqual(len(store.list_all()), 10)  # Explicit admin inventory.

    def test_identity_and_global_inheritance_are_independent(self) -> None:
        store = MemoryStore(self.db_path)
        shared = store.add("Python project convention", project_id="a")
        mine = store.add("Python personal convention", project_id="a", user_id="alice")
        store.add("Python other convention", project_id="a", user_id="bob")
        self.assertEqual({item.id for item in store.search("Python", project_id="a", user_id="alice")}, {shared.id, mine.id})
        self.assertEqual(store.search("Python", project_id="a", user_id="alice", include_global=False), [mine])

    def test_secrets_in_every_persisted_payload_are_rejected(self) -> None:
        store = MemoryStore(self.db_path)
        credential = "api_key=sk-example-secret-value"
        for field in ("summary", "source_type", "source_reference", "project_id", "user_id", "session_id", "task_id", "agent_id"):
            with self.subTest(field=field), self.assertRaises(SensitiveMemoryError):
                store.add("Safe fact", **{field: credential})
        for metadata in ({"nested": [{"password": "tiny"}]}, {"nested": {"note": credential}}, {"secret": "credential"}):
            with self.subTest(metadata=metadata), self.assertRaises(SensitiveMemoryError):
                store.add("Safe fact", metadata=metadata)
        self.assertEqual(store.list_all(), [])
        saved = store.add("Safe fact")
        for update in ({"summary": credential}, {"metadata": {"token_note": credential}}):
            with self.assertRaises(SensitiveMemoryError):
                store.update(saved.id, **update)
        self.assertEqual(store.get(saved.id), saved)

    def test_redaction_and_validation_share_credential_patterns(self) -> None:
        store = MemoryStore(self.db_path)
        samples = [
            "refresh_token=small", "api_key='short'", "password=abc",
            "Authorization: abcdef", "Bearer short-token",
            "github_pat_1234567890abcdefgh", "AKIA1234567890ABCDEF",
            "postgresql://user:p4ssword@db.example.test/database",
            "-----BEGIN RSA PRIVATE KEY-----\nSECRET BODY\n-----END RSA PRIVATE KEY-----",
            "-----BEGIN PRIVATE KEY-----\nINCOMPLETE SECRET BODY",
        ]
        for raw in samples:
            with self.subTest(raw=raw):
                with self.assertRaises(SensitiveMemoryError):
                    store.add(raw)
                clean = redact_sensitive_text(raw)
                self.assertIn("[REDACTED]", clean)
                self.assertNotEqual(raw, clean)
                self.assertEqual(redact_sensitive_text(clean), clean)
                self.assertEqual(store.add(clean).content, clean)
        redacted_json = redact_sensitive_text('{"password": "spaces in secret"}')
        self.assertEqual(json.loads(redacted_json), {"password": "[REDACTED]"})
        placeholder = store.add("Credential removed", metadata={"password": "[REDACTED]"}, summary='api_key="[REDACTED]"')
        self.assertEqual(placeholder.metadata["password"], "[REDACTED]")

    def test_expired_duplicate_is_renewed_with_new_provenance(self) -> None:
        store = MemoryStore(self.db_path)
        old = store.add("Python dependency constraint", valid_until=datetime.now(timezone.utc) - timedelta(days=1), source_reference="old")
        current = store.add("Python dependency constraint", source_reference="confirmed-again")
        self.assertNotEqual(old.id, current.id)
        self.assertEqual(store.search("Python"), [current])
        self.assertEqual(current.source_reference, "confirmed-again")

    def test_supersede_cannot_change_scope_or_duplicate_or_self_link(self) -> None:
        store = MemoryStore(self.db_path)
        old = store.add("Timeout thirty", project_id="a", user_id="alice")
        other = store.add("Timeout ninety", project_id="a", user_id="alice")
        for fields in ({"project_id": "b"}, {"user_id": None}, {"session_id": "new"}, {"supersedes_id": other.id}, {"status": "expired"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                store.supersede(old.id, "Timeout sixty", **fields)
        with self.assertRaises(ValueError):
            store.supersede(old.id, old.content)
        with self.assertRaises(ValueError):
            store.supersede(old.id, other.content)
        with self.assertRaises(SensitiveMemoryError):
            store.supersede(old.id, "Timeout sixty", metadata={"password": "secret"})
        self.assertEqual(store.get(old.id), old)
        self.assertEqual(len(store.list_all()), 2)

    def test_supersede_rolls_back_insert_when_history_update_fails(self) -> None:
        store = MemoryStore(self.db_path)
        old = store.add("Original Python requirement")
        with store._connect(write=True) as connection:
            connection.execute("""CREATE TRIGGER simulate_update_failure
                BEFORE UPDATE OF status ON memories WHEN new.status = 'superseded'
                BEGIN SELECT RAISE(ABORT, 'simulated failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            store.supersede(old.id, "Replacement Python requirement")
        self.assertEqual(store.list_all(), [old])
        self.assertEqual(store.search("Replacement"), [])

    def test_direct_add_supersession_is_atomic_and_scope_checked(self) -> None:
        store = MemoryStore(self.db_path)
        old = store.add("Old Python requirement", project_id="a")
        with self.assertRaises(ValueError):
            store.add("New Python requirement", project_id="b", supersedes_id=old.id)
        self.assertEqual(store.get(old.id).status, "active")
        new = store.add("New Python requirement", project_id="a", supersedes_id=old.id)
        self.assertEqual(new.supersedes_id, old.id)
        self.assertEqual(store.get(old.id).status, "superseded")

    def test_concurrent_duplicate_writers_insert_one_row(self) -> None:
        store = MemoryStore(self.db_path)
        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(lambda _: store.add("Concurrent SQLite fact", project_id="a"), range(24)))
        self.assertEqual(len({item.id for item in records}), 1)
        self.assertEqual(len(store.list_all()), 1)

    def test_concurrent_supersession_has_only_one_successor(self) -> None:
        store = MemoryStore(self.db_path)
        old = store.add("Original requirement")
        def replace(index: int):
            try:
                return store.supersede(old.id, f"Replacement {index}")
            except ValueError:
                return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            replacements = list(pool.map(replace, range(4)))
        self.assertEqual(sum(item is not None for item in replacements), 1)
        self.assertEqual(len(store.list_all(status="active")), 1)
        self.assertEqual(store.get(old.id).status, "superseded")

    def test_connections_close_and_schema_uses_wal(self) -> None:
        store = MemoryStore(self.db_path)
        with store._connect() as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(connection.execute("PRAGMA busy_timeout").fetchone()[0], 15000)
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_updates_cannot_create_active_duplicates(self) -> None:
        store = MemoryStore(self.db_path)
        first = store.add("Python 3.12")
        second = store.add("Python 3.13")
        with self.assertRaises(ValueError):
            store.update(second.id, content=first.content)
        self.assertEqual(store.get(second.id), second)

    def test_code_punctuation_is_not_deduplicated_away(self) -> None:
        store = MemoryStore(self.db_path)
        cpp = store.add("Language C++")
        csharp = store.add("Language C#")
        self.assertEqual({item.id for item in store.search("Language")}, {cpp.id, csharp.id})
        self.assertEqual(store.search('" OR NOT * ^'), [])

    def test_common_question_words_do_not_create_false_matches(self) -> None:
        store = MemoryStore(self.db_path)
        store.add("The user interface is dark")
        self.assertEqual(store.search("What is the database?"), [])

    def test_long_request_keeps_error_identifiers_at_the_end(self) -> None:
        store = MemoryStore(self.db_path)
        error = store.add("EADDRINUSE means inspect the listener before retrying")
        prompt = " ".join(f"backgroundword{index}" for index in range(100))
        self.assertEqual(store.search(prompt + ". First resolve EADDRINUSE."), [error])
        self.assertLessEqual(len(store._fts_query(prompt).split(" OR ")), 48)

    def test_late_paths_and_symbols_survive_large_prose_prompt(self) -> None:
        store = MemoryStore(self.db_path)
        path = store.add("The serializer lives in src/recovery_handler.py")
        # Unique lowercase prose is deliberately less informative than a path.
        prose = " ".join("word" + chr(97 + index // 26) + chr(97 + index % 26) for index in range(100))
        self.assertEqual(store.search(prose + " Investigate src/recovery_handler.py"), [path])
        bounded = store._fts_query(prose + ' "; NOT * OR NEAR(foo) -- ' + "x" * 1000)
        self.assertLess(len(bounded), 48 * 135)
        store.search(bounded)  # User punctuation cannot inject FTS operators.


class FakeEmbeddingProvider:
    fingerprint = "fake-semantic-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise TimeoutError("simulated provider failure")
        return [[1.0, 0.0] if any(word in text.lower() for word in ("pytest", "verification", "testing")) else [0.0, 1.0] for text in texts]


class HybridMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "memory.db"
        self.provider = FakeEmbeddingProvider()
        self.store = MemoryStore(self.path, embedding_provider=self.provider)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_semantic_paraphrase_and_scope_before_provider(self) -> None:
        eligible = self.store.add("Run pytest before publishing", project_id="a")
        self.store.add("Private pytest secret plan", project_id="a", user_id="bob")
        self.store.add("Other pytest workspace", project_id="b")
        self.store.add("Expired pytest command", project_id="a", valid_until=datetime.now(timezone.utc) - timedelta(seconds=1))
        self.store.add("Unrelated colour theme", project_id="a")
        hits = self.store.retrieve("verification", project_id="a")
        self.assertEqual([hit.memory_id for hit in hits], [eligible.id])
        self.assertIn("semantic cosine=", hits[0].reason_retrieved)
        sent = [text for batch in self.provider.calls for text in batch]
        self.assertNotIn("Private pytest secret plan", sent)
        self.assertNotIn("Other pytest workspace", sent)
        self.assertNotIn("Expired pytest command", sent)
        self.assertEqual(hits[0].record.project_id, "a")

    def test_vectors_persist_and_invalidate_on_edit_or_model_change(self) -> None:
        record = self.store.add("Run pytest")
        self.store.retrieve("verification")
        self.provider.calls.clear()
        restarted = MemoryStore(self.path, embedding_provider=self.provider)
        restarted.retrieve("verification")
        self.assertEqual(self.provider.calls, [["verification"]])
        restarted.update(record.id, content="Run pytest with coverage")
        self.provider.calls.clear()
        restarted.retrieve("verification")
        self.assertIn(["Run pytest with coverage"], self.provider.calls)
        self.provider.fingerprint = "fake-semantic-v2"
        self.provider.calls.clear()
        restarted.retrieve("verification")
        self.assertIn(["Run pytest with coverage"], self.provider.calls)
        restarted.delete(record.id)
        with restarted._connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0], 0)

    def test_provider_failure_gracefully_uses_fts(self) -> None:
        record = self.store.add("Run pytest")
        self.provider.fail = True
        with self.assertLogs("memory.memory", level="WARNING"):
            hits = self.store.retrieve("pytest")
        self.assertEqual([hit.memory_id for hit in hits], [record.id])
        self.assertNotIn("semantic", hits[0].reason_retrieved)

    def test_corrupt_provider_vectors_fall_back_and_are_not_cached(self) -> None:
        record = self.store.add("Run pytest")
        with patch.object(self.provider, "embed", return_value=[[float("nan"), 0.0]]):
            with self.assertLogs("memory.memory", level="WARNING"):
                hits = self.store.retrieve("pytest")
        self.assertEqual([hit.memory_id for hit in hits], [record.id])
        with self.store._connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0], 0)

    def test_sensitive_query_is_never_sent_to_provider(self) -> None:
        self.store.add("Run pytest")
        with self.assertLogs("memory.memory", level="WARNING"):
            self.store.retrieve("pytest api_key=sk-example-secret-value")
        self.assertEqual(self.provider.calls, [])

    def test_empty_scope_does_not_contact_provider(self) -> None:
        self.store.add("Run pytest", project_id="private")
        self.assertEqual(self.store.retrieve("verification", project_id="other"), [])
        self.assertEqual(self.provider.calls, [])

    def test_semantic_work_is_bounded(self) -> None:
        for index in range(8):
            self.store.add(f"Run pytest suite {index}")
        bounded = MemoryStore(self.path, embedding_provider=self.provider, semantic_candidate_limit=3)
        bounded.retrieve("verification")
        self.assertEqual(sum(len(batch) for batch in self.provider.calls), 4)

    def test_cached_match_survives_more_than_256_newer_peers(self) -> None:
        target = self.store.add("Run pytest before shipping")
        self.assertEqual(self.store.search("verification"), [target])
        for index in range(256):
            self.store.add(f"Colour layout alternative {index}")
        self.provider.calls.clear()
        self.assertEqual(self.store.search("verification"), [target])
        self.assertEqual(sum(len(batch) for batch in self.provider.calls), 256)
        self.provider.calls.clear()
        self.assertEqual(self.store.search("verification"), [target])
        self.assertEqual(self.provider.calls, [])

    def test_query_failure_does_not_pay_for_document_embeddings(self) -> None:
        for index in range(64):
            self.store.add(f"Run pytest suite {index}")
        self.provider.fail = True
        with self.assertLogs("memory.memory", level="WARNING"):
            for _ in range(3):
                self.store.retrieve("verification")
        self.assertEqual(self.provider.calls, [["verification"]] * 3)
        with self.store._connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0], 0)

    def test_completed_document_batches_survive_later_provider_failure(self) -> None:
        for index in range(64):
            self.store.add(f"Run pytest suite {index}")
        original_embed = self.provider.embed
        document_calls = 0
        def fail_second_document_batch(texts):
            nonlocal document_calls
            if len(texts) > 1:
                document_calls += 1
                if document_calls == 2:
                    raise TimeoutError("Second batch unavailable")
            return original_embed(texts)
        with patch.object(self.provider, "embed", side_effect=fail_second_document_batch):
            with self.assertLogs("memory.memory", level="WARNING"):
                hits = self.store.retrieve("verification")
        self.assertEqual(len(hits), 3)  # Successful work is useful immediately.
        completed = set(self.provider.calls[1])
        with self.store._connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0], 32)
        self.provider.calls.clear()
        self.store.retrieve("verification")
        sent = [text for batch in self.provider.calls for text in batch]
        self.assertEqual(len(sent), 32)
        self.assertTrue(completed.isdisjoint(sent))
        self.assertNotIn("verification", sent)

    def test_repeated_query_uses_bounded_process_cache_and_model_key(self) -> None:
        self.store.add("Run pytest")
        self.store.retrieve("verification")
        self.provider.calls.clear()
        for _ in range(5):
            self.store.retrieve("verification")
        self.assertEqual(self.provider.calls, [])
        for index in range(33):
            self.store.retrieve(f"verification {index}")
        self.provider.calls.clear()
        self.store.retrieve("verification")
        self.assertEqual(self.provider.calls, [["verification"]])
        self.provider.fingerprint = "new-embedding-model"
        self.provider.calls.clear()
        self.store.retrieve("verification")
        self.assertEqual(self.provider.calls, [["verification"], ["Run pytest"]])

    def test_provider_calls_never_hold_a_database_write_lock(self) -> None:
        self.store.add("Run pytest")
        original_embed = self.provider.embed
        def checking_embed(texts):
            connection = sqlite3.connect(self.path, timeout=0.01)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.rollback()
            finally:
                connection.close()
            return original_embed(texts)
        with patch.object(self.provider, "embed", side_effect=checking_embed):
            self.assertEqual(len(self.store.retrieve("verification")), 1)

    def test_failed_ingestion_preserves_older_cached_semantic_matches(self) -> None:
        target = self.store.add("Run pytest before shipping")
        self.store.retrieve("verification")
        self.store.add("New unrelated colour preference")
        self.provider.fail = True
        with self.assertLogs("memory.memory", level="WARNING"):
            self.assertEqual(self.store.search("verification"), [target])

    def test_changed_content_during_embedding_does_not_cache_stale_vectors(self) -> None:
        target = self.store.add("Run pytest")
        original_embed = self.provider.embed
        def changing_embed(texts):
            if "Run pytest" in texts:
                self.store.update(target.id, content="Use the blue colour scheme")
            return original_embed(texts)
        with patch.object(self.provider, "embed", side_effect=changing_embed):
            self.assertEqual(self.store.search("verification"), [])
        with self.store._connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0], 0)

    def test_concurrent_deletion_during_embedding_does_not_restore_memory(self) -> None:
        record = self.store.add("Run pytest")
        original_embed = self.provider.embed
        def deleting_embed(texts):
            self.store.delete(record.id)
            return original_embed(texts)
        with patch.object(self.provider, "embed", side_effect=deleting_embed):
            self.assertEqual(self.store.retrieve("verification"), [])
        self.assertIsNone(self.store.get(record.id))
        with self.store._connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0], 0)

    def test_openrouter_request_contract_without_network(self) -> None:
        provider = OpenRouterEmbeddingProvider(api_key="test-key", dimensions=2)
        response = io.BytesIO(json.dumps({"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]}).encode())
        with patch("memory.embeddings.request.urlopen", return_value=response) as mocked:
            vectors = provider.embed(["first", "second"])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        sent = mocked.call_args.args[0]
        self.assertEqual(sent.full_url, "https://openrouter.ai/api/v1/embeddings")
        self.assertEqual(json.loads(sent.data)["input"], ["first", "second"])
        self.assertEqual(json.loads(sent.data)["dimensions"], 2)
        self.assertEqual(sent.get_header("Authorization"), "Bearer test-key")
        self.assertNotIn("test-key", repr(provider))
        self.assertNotIn("test-key", provider.fingerprint)

    def test_openrouter_rejects_duplicate_response_indexes(self) -> None:
        provider = OpenRouterEmbeddingProvider(api_key="test-key")
        response = io.BytesIO(json.dumps({"data": [
            {"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]},
        ]}).encode())
        with patch("memory.embeddings.request.urlopen", return_value=response), self.assertRaises(ValueError):
            provider.embed(["first", "second"])


if __name__ == "__main__":
    unittest.main()
