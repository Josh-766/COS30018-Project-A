"""Persistent, scoped long-term memory using the native LangGraph store."""

from pathlib import Path
import tempfile
import unittest

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from memory import (
    MemoryContext,
    SensitiveMemoryError,
    forget_memory,
    list_memories,
    open_memory,
    save_memory,
    search_memories,
    thread_config,
)


class MemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.db_path = Path(self.temporary_directory.name) / "memory.db"
        self.context = MemoryContext(project_id="project-a", user_id="alice")

    def test_native_store_and_checkpointer_survive_reopening_database(self) -> None:
        with open_memory(self.db_path) as (checkpointer, store):
            self.assertIsInstance(checkpointer, SqliteSaver)
            self.assertIsInstance(store, SqliteStore)
            saved = save_memory(store, self.context, "API requests need a 60-second timeout")

        with open_memory(self.db_path) as (_, restarted_store):
            self.assertEqual(search_memories(restarted_store, self.context, "timeout"), [saved])
            self.assertEqual(list_memories(restarted_store, self.context), [saved])

    def test_duplicate_has_stable_id_without_creating_a_second_record(self) -> None:
        with open_memory(self.db_path) as (_, store):
            first = save_memory(store, self.context, "Use SQLite")
            duplicate = save_memory(store, self.context, "  Use SQLite  ")
            self.assertEqual(duplicate, first)
            self.assertEqual(list_memories(store, self.context), [first])

        with open_memory(self.db_path) as (_, store):
            restarted_duplicate = save_memory(store, self.context, "Use SQLite")
            self.assertEqual(restarted_duplicate, first)
            self.assertEqual(list_memories(store, self.context), [first])

    def test_memories_are_isolated_by_project_and_user(self) -> None:
        other_project = MemoryContext(project_id="project-b", user_id="alice")
        other_user = MemoryContext(project_id="project-a", user_id="bob")
        with open_memory(self.db_path) as (_, store):
            own = save_memory(store, self.context, "Project uses Python")
            project_memory = save_memory(store, other_project, "Project uses Rust")
            user_memory = save_memory(store, other_user, "Project uses Java")

            for context, expected in (
                (self.context, own),
                (other_project, project_memory),
                (other_user, user_memory),
            ):
                with self.subTest(context=context):
                    self.assertEqual(search_memories(store, context, "project"), [expected])
                    self.assertEqual(list_memories(store, context), [expected])

            self.assertFalse(forget_memory(store, other_project, own["id"]))
            self.assertFalse(forget_memory(store, other_user, own["id"]))
            self.assertEqual(list_memories(store, self.context), [own])

    def test_keyword_search_filters_irrelevant_memories_and_obeys_limit(self) -> None:
        with open_memory(self.db_path) as (_, store):
            matching = [
                save_memory(store, self.context, f"Python setting number {index}")
                for index in range(8)
            ]
            save_memory(store, self.context, "The interface uses a dark palette")

            results = search_memories(store, self.context, "PYTHON", limit=3)

            self.assertEqual(len(results), 3)
            self.assertTrue(all(record in matching for record in results))
            self.assertEqual(search_memories(store, self.context, "unrelatedword"), [])
            self.assertEqual(search_memories(store, self.context, ""), [])
            self.assertEqual(search_memories(store, self.context, "Python", limit=0), [])

    def test_list_and_search_read_beyond_the_first_store_page(self) -> None:
        with open_memory(self.db_path) as (_, store):
            saved = [
                save_memory(store, self.context, f"Unique marker{index:04d}")
                for index in range(205)
            ]

            self.assertEqual(
                {record["id"] for record in list_memories(store, self.context)},
                {record["id"] for record in saved},
            )
            # Exercise both ends regardless of the store's ordering.
            for index in (0, 204):
                with self.subTest(index=index):
                    self.assertEqual(
                        search_memories(store, self.context, f"marker{index:04d}"),
                        [saved[index]],
                    )

    def test_forget_is_persistent_and_reports_an_absent_record(self) -> None:
        with open_memory(self.db_path) as (_, store):
            saved = save_memory(store, self.context, "Use SQLite for memory")
            self.assertTrue(forget_memory(store, self.context, saved["id"]))
            self.assertFalse(forget_memory(store, self.context, saved["id"]))
            self.assertEqual(search_memories(store, self.context, "SQLite"), [])

        with open_memory(self.db_path) as (_, store):
            self.assertEqual(list_memories(store, self.context), [])

    def test_empty_oversized_and_sensitive_content_are_not_written(self) -> None:
        with open_memory(self.db_path) as (_, store):
            for content in ("", "   ", "x" * 2001):
                with self.subTest(content_length=len(content)):
                    with self.assertRaises(ValueError):
                        save_memory(store, self.context, content)

            for content in (
                "api_key=sk-example-secret-value",
                "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
                "-----BEGIN PRIVATE KEY-----",
            ):
                with self.subTest(content=content):
                    with self.assertRaises(SensitiveMemoryError):
                        save_memory(store, self.context, content)

            self.assertEqual(list_memories(store, self.context), [])
            self.assertEqual(save_memory(store, self.context, "x" * 2000)["content"], "x" * 2000)

    def test_checkpoint_thread_keys_isolate_project_user_and_thread(self) -> None:
        configs = [
            thread_config(self.context, "thread-one"),
            thread_config(self.context, "thread-two"),
            thread_config(MemoryContext(project_id="project-b", user_id="alice"), "thread-one"),
            thread_config(MemoryContext(project_id="project-a", user_id="bob"), "thread-one"),
        ]
        self.assertEqual(len({config["configurable"]["thread_id"] for config in configs}), 4)
        self.assertEqual(thread_config(self.context, "thread-one"), configs[0])


if __name__ == "__main__":
    unittest.main()
