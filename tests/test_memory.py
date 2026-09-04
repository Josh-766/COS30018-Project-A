import tempfile
from pathlib import Path
import unittest

from memory import MemoryStore, get_relevant_memories


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


if __name__ == "__main__":
    unittest.main()
