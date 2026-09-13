import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import unittest

from memory import MemoryStore, SensitiveMemoryError, get_relevant_memories


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

        hits = store.retrieve("Python version", max_tokens=3)

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


if __name__ == "__main__":
    unittest.main()
