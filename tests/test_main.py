"""CLI commands exercised with native checkpoints, without services or credentials."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_roles.graph import create_coder_graph
from main import chat
from memory import MemoryContext, list_memories, open_memory, thread_config


class ChatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project = Path(self.temporary_directory.name)
        self.db_path = self.project / "memory.db"
        self.context = MemoryContext(project_id="project-a")
        self.sandbox = Mock()

    def graph(self, checkpointer, store):
        return create_coder_graph(
            checkpointer=checkpointer,
            store=store,
            sandbox=self.sandbox,
            project=self.project,
        )

    def test_manual_memory_commands_do_not_call_the_model(self) -> None:
        commands = iter(["/remember Use SQLite", "/memories", "FORGET", "/memories", "exit"])
        output = StringIO()
        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)

            def next_input(prompt):
                command = next(commands)
                if command == "FORGET":
                    saved = list_memories(store, self.context)
                    self.assertEqual(len(saved), 1)
                    return f"/forget {saved[0]['id']}"
                return command

            with patch("builtins.input", side_effect=next_input):
                with patch("agent_roles.graph.send_to_coder") as model, redirect_stdout(output):
                    chat(graph, store, self.context, "original-thread")
            model.assert_not_called()
            self.assertEqual(list_memories(store, self.context), [])
            self.assertFalse(graph.get_state(thread_config(self.context, "original-thread")).values)

        self.assertIn("Use SQLite", output.getvalue())
        self.assertIn("deleted", output.getvalue())
        self.assertIn("no saved memories", output.getvalue())
        self.sandbox._execute.assert_not_called()

    def test_new_starts_a_fresh_thread_and_resume_restores_the_selected_thread(self) -> None:
        commands = [
            "First request",
            "/new",
            "A separate request",
            "/resume original-thread",
            "Continue the first request",
            "exit",
        ]
        reply = {"message": {"role": "assistant", "content": "Done."}}
        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            with patch("builtins.input", side_effect=commands):
                with patch("main.uuid4", return_value=SimpleNamespace(hex="new-thread")):
                    with patch("agent_roles.graph.send_to_coder", return_value=reply) as model:
                        with redirect_stdout(StringIO()):
                            chat(graph, store, self.context, "original-thread")

            contexts = [call.kwargs["context"] for call in model.call_args_list]
            self.assertEqual(contexts[0], [{"role": "user", "content": "First request"}])
            self.assertEqual(contexts[1], [{"role": "user", "content": "A separate request"}])
            self.assertEqual(
                contexts[2],
                [
                    {"role": "user", "content": "First request"},
                    reply["message"],
                    {"role": "user", "content": "Continue the first request"},
                ],
            )
            new_thread = graph.get_state(thread_config(self.context, "new-thread")).values
            self.assertEqual(len(new_thread["messages"]), 2)

    def test_retry_resumes_failed_turn_without_duplicating_the_user_message(self) -> None:
        output = StringIO()
        commands = ["Initial request", "Do not append this", "/retry", "/retry", "exit"]
        reply = {"message": {"role": "assistant", "content": "Recovered."}}
        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            with patch("builtins.input", side_effect=commands):
                with patch(
                    "agent_roles.graph.send_to_coder",
                    side_effect=[RuntimeError("temporary model failure"), reply],
                ) as model:
                    with redirect_stdout(output):
                        chat(graph, store, self.context, "original-thread")

            self.assertEqual(model.call_count, 2)
            expected_input = [{"role": "user", "content": "Initial request"}]
            self.assertEqual(model.call_args_list[0].kwargs["context"], expected_input)
            self.assertEqual(model.call_args_list[1].kwargs["context"], expected_input)
            state = graph.get_state(thread_config(self.context, "original-thread"))
            self.assertEqual(state.values["messages"], expected_input + [reply["message"]])
            self.assertFalse(state.next)

        self.assertIn("temporary model failure", output.getvalue())
        self.assertIn("unfinished turn", output.getvalue())
        self.assertIn("There is no unfinished turn to retry", output.getvalue())
        self.assertIn("Recovered.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
