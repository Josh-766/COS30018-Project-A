"""Exercise the graph with a real SQLite database and a scripted model."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_roles.graph import create_coder_graph
from memory import (
    MemoryContext,
    list_memories,
    open_memory,
    save_memory,
    thread_config,
)
from scripts.tools.tool_registry import list_tools


def assistant(content="Done", *, tool_calls=None, **extra):
    message = {"role": "assistant", "content": content, **extra}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message}


def tool_call(name, arguments, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class AgentMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project = Path(self.temporary_directory.name)
        self.db_path = self.project / "memory.db"
        self.context = MemoryContext(project_id="project-a", user_id="alice")
        self.sandbox = Mock()
        self.sandbox._execute.return_value = SimpleNamespace(
            exit_code=0, stdout=b"print('hello')\n", stderr=b""
        )

    def graph(self, checkpointer, store):
        return create_coder_graph(
            checkpointer=checkpointer,
            store=store,
            sandbox=self.sandbox,
            project=self.project,
        )

    def invoke(self, graph, text, thread="thread-one", context=None):
        context = context or self.context
        return graph.invoke(
            {"messages": [{"role": "user", "content": text}]},
            thread_config(context, thread),
            context=context,
        )

    def test_short_term_history_survives_a_database_and_graph_restart(self) -> None:
        first_reply = assistant(
            "I will use Python.",
            reasoning_details=[
                {"type": "reasoning.encrypted", "data": "opaque-provider-payload", "index": 0}
            ],
        )
        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            with patch("agent_roles.graph.send_to_coder", return_value=first_reply):
                first = self.invoke(graph, "Use Python.")

        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            with patch("agent_roles.graph.send_to_coder", return_value=assistant("Yes.")) as model:
                resumed = self.invoke(graph, "Do you remember?")

            expected_context = first["messages"] + [{"role": "user", "content": "Do you remember?"}]
            self.assertEqual(model.call_args.kwargs["context"], expected_context)
            self.assertEqual(resumed["messages"][:-1], expected_context)
            self.assertEqual(resumed["messages"][1], first_reply["message"])
            self.assertEqual(
                graph.get_state(thread_config(self.context, "thread-one")).values["messages"],
                resumed["messages"],
            )

    def test_new_threads_projects_and_users_start_with_empty_history(self) -> None:
        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            with patch("agent_roles.graph.send_to_coder", return_value=assistant("Recorded.")):
                self.invoke(graph, "Private first conversation")

            for context, thread in (
                (self.context, "thread-two"),
                (MemoryContext(project_id="project-b", user_id="alice"), "thread-one"),
                (MemoryContext(project_id="project-a", user_id="bob"), "thread-one"),
            ):
                with self.subTest(context=context, thread=thread):
                    with patch("agent_roles.graph.send_to_coder", return_value=assistant()) as model:
                        result = self.invoke(graph, "A fresh conversation", thread, context)
                    self.assertEqual(
                        model.call_args.kwargs["context"],
                        [{"role": "user", "content": "A fresh conversation"}],
                    )
                    self.assertEqual(len(result["messages"]), 2)

    def test_durable_memories_are_only_retrieved_when_the_agent_calls_a_tool(self) -> None:
        with open_memory(self.db_path) as (checkpointer, store):
            saved = save_memory(store, self.context, "Use the durable-memory-sentinel compiler")
            graph = self.graph(checkpointer, store)
            with patch.object(store, "search", wraps=store.search) as search:
                with patch("agent_roles.graph.send_to_coder", return_value=assistant()) as model:
                    self.invoke(graph, "Which compiler?")
                search.assert_not_called()
                self.assertNotIn(saved["content"], json.dumps(model.call_args.kwargs))
            self.assertEqual(list_memories(store, self.context), [saved])

    def test_agent_can_save_search_and_forget_memories_in_one_tool_loop(self) -> None:
        observed = []
        saved = {}

        def model_response(text, *, context, **kwargs):
            observed.append(deepcopy(context))
            step = len(observed)
            if step == 1:
                return assistant(None, tool_calls=[tool_call("save_memory", {"content": "Use SQLite"}, "save")])
            if step == 2:
                saved.update(json.loads(context[-1]["content"]))
                return assistant(None, tool_calls=[tool_call("search_memories", {"query": "SQLite"}, "search")])
            if step == 3:
                self.assertEqual(json.loads(context[-1]["content"]), [saved])
                return assistant(None, tool_calls=[tool_call("forget_memory", {"memory_id": saved["id"]}, "forget")])
            return assistant("Saved, found, and forgotten.")

        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            with patch("agent_roles.graph.send_to_coder", side_effect=model_response):
                result = self.invoke(graph, "Remember SQLite, find it, then forget it.")
            self.assertEqual(list_memories(store, self.context), [])

        self.assertEqual(saved["content"], "Use SQLite")
        self.assertEqual(len(observed), 4)
        outputs = [message for message in result["messages"] if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in outputs], ["save", "search", "forget"])
        self.assertTrue(all(isinstance(message["content"], str) for message in outputs))
        self.assertIs(json.loads(outputs[-1]["content"]), True)
        self.assertEqual(result["messages"][-1]["content"], "Saved, found, and forgotten.")
        self.sandbox._execute.assert_not_called()

    def test_agent_finds_long_term_memory_from_a_different_thread_after_restart(self) -> None:
        with open_memory(self.db_path) as (checkpointer, store):
            with patch(
                "agent_roles.graph.send_to_coder",
                side_effect=[
                    assistant(None, tool_calls=[tool_call("save_memory", {"content": "Use Python 3.12"})]),
                    assistant("Saved."),
                ],
            ):
                first = self.invoke(self.graph(checkpointer, store), "Remember the Python version.")
            saved = json.loads(next(message["content"] for message in first["messages"] if message["role"] == "tool"))

        with open_memory(self.db_path) as (checkpointer, store):
            with patch(
                "agent_roles.graph.send_to_coder",
                side_effect=[
                    assistant(None, tool_calls=[tool_call("search_memories", {"query": "Python"})]),
                    assistant("Python 3.12."),
                ],
            ) as model:
                result = self.invoke(self.graph(checkpointer, store), "What version?", "thread-two")
            self.assertEqual(model.call_args_list[0].kwargs["context"], [{"role": "user", "content": "What version?"}])
            retrieved = next(message for message in result["messages"] if message["role"] == "tool")
            self.assertEqual(json.loads(retrieved["content"]), [saved])

    def test_malformed_tool_arguments_produce_errors_and_allow_the_agent_to_continue(self) -> None:
        malformed_json = tool_call("save_memory", {})
        malformed_json["function"]["arguments"] = "{not valid JSON"
        calls = [
            malformed_json,
            tool_call("save_memory", {}),
            tool_call("save_memory", {"content": 42}),
            tool_call("save_memory", {"content": "Use SQLite", "user_id": "bob"}),
            tool_call("save_memory", ["Use SQLite"]),
            tool_call("save_memory", {"content": "password=example-secret-value"}),
            tool_call("not_a_tool", {}),
        ]
        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            for index, call in enumerate(calls):
                with self.subTest(call=call):
                    with patch(
                        "agent_roles.graph.send_to_coder",
                        side_effect=[assistant(None, tool_calls=[call]), assistant("Recovered.")],
                    ):
                        result = self.invoke(graph, "Try this tool", f"bad-call-{index}")
                    output = next(message for message in result["messages"] if message["role"] == "tool")
                    self.assertEqual(output["tool_call_id"], call["id"])
                    self.assertIn("error", json.loads(output["content"]))
                    self.assertEqual(result["messages"][-1]["content"], "Recovered.")
            self.assertEqual(list_memories(store, self.context), [])
        self.sandbox._execute.assert_not_called()

    def test_existing_sandbox_tools_still_dispatch_and_sync_to_host(self) -> None:
        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            with patch(
                "agent_roles.graph.send_to_coder",
                side_effect=[
                    assistant(None, tool_calls=[tool_call("read_file", {"path": "main.py"})]),
                    assistant("Read the file."),
                ],
            ):
                result = self.invoke(graph, "Read main.py")

        self.sandbox._execute.assert_called_once_with(["cat", "--", "/home/user/project/main.py"])
        self.sandbox.save_to_host.assert_called_once_with(self.project)
        output = next(message for message in result["messages"] if message["role"] == "tool")
        self.assertEqual(json.loads(output["content"]), {"content": "print('hello')\n", "truncated": False})

    def test_completed_command_is_not_replayed_after_sync_failure_and_restart(self) -> None:
        call = tool_call("run_command", {"command": "printf completed >> events.log"})
        first_sandbox = self.sandbox
        first_sandbox.save_to_host.side_effect = RuntimeError("host sync failed")
        config = thread_config(self.context, "thread-one")
        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            with patch(
                "agent_roles.graph.send_to_coder",
                return_value=assistant(None, tool_calls=[call]),
            ) as model:
                with self.assertRaisesRegex(RuntimeError, "host sync failed"):
                    self.invoke(graph, "Append one event.")
            model.assert_called_once()
            first_sandbox._execute.assert_called_once()
            self.assertTrue(graph.get_state(config).next)

        # Rebuild the graph and connections, as a restarted application does.
        self.sandbox = Mock()
        with open_memory(self.db_path) as (checkpointer, store):
            graph = self.graph(checkpointer, store)
            with patch("agent_roles.graph.send_to_coder", return_value=assistant("Completed.")):
                result = graph.invoke(None, config, context=self.context)
            self.assertFalse(graph.get_state(config).next)

        self.sandbox._execute.assert_not_called()
        self.sandbox.save_to_host.assert_called_once_with(self.project)
        self.assertEqual(
            [message["role"] for message in result["messages"]],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(result["messages"][2]["tool_call_id"], call["id"])
        self.assertEqual(json.loads(result["messages"][2]["content"])["exit_code"], 0)

    def test_real_coder_adapter_sends_checkpointed_raw_history_and_memory_tools(self) -> None:
        tool_reply = assistant(
            None,
            tool_calls=[tool_call("save_memory", {"content": "Use SQLite"}, "remember-sqlite")],
            reasoning_details=[
                {"type": "reasoning.encrypted", "data": "opaque-provider-payload", "index": 0}
            ],
        )["message"]
        final_reply = assistant("Saved SQLite.")["message"]
        resumed_reply = assistant("Yes, SQLite.")["message"]
        responses = []
        for message in (tool_reply, final_reply, resumed_reply):
            response = Mock()
            response.json.return_value = {
                "choices": [{
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }]
            }
            responses.append(response)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only-api-key"}):
            with patch("agent_roles.coder.requests.post", side_effect=responses) as post:
                with open_memory(self.db_path) as (checkpointer, store):
                    first = self.invoke(self.graph(checkpointer, store), "Remember SQLite.")
                    self.assertEqual(len(list_memories(store, self.context)), 1)
                with open_memory(self.db_path) as (checkpointer, store):
                    resumed = self.invoke(self.graph(checkpointer, store), "Do you remember?")

        self.assertEqual(post.call_count, 3)
        payloads = [call.kwargs["json"] for call in post.call_args_list]
        self.assertEqual(payloads[0]["messages"][1:], [{"role": "user", "content": "Remember SQLite."}])
        self.assertEqual(payloads[1]["messages"][1:], first["messages"][:-1])
        expected_history = first["messages"] + [{"role": "user", "content": "Do you remember?"}]
        self.assertEqual(payloads[2]["messages"][1:], expected_history)
        self.assertEqual(resumed["messages"], expected_history + [resumed_reply])
        self.assertEqual(resumed["messages"][1], tool_reply)
        for payload in payloads:
            self.assertEqual(sum(message["role"] == "system" for message in payload["messages"]), 1)
            names = {tool["function"]["name"] for tool in payload["tools"]}
            self.assertTrue({"save_memory", "search_memories", "forget_memory"}.issubset(names))
        for response in responses:
            response.raise_for_status.assert_called_once()

    def test_model_tool_schemas_expose_memory_actions_without_scope_overrides(self) -> None:
        definitions = {tool["function"]["name"]: tool["function"] for tool in list_tools()}
        self.assertTrue({"run_command", "read_file", "write_file", "list_files"}.issubset(definitions))
        for name, field in (
            ("save_memory", "content"),
            ("search_memories", "query"),
            ("forget_memory", "memory_id"),
        ):
            with self.subTest(tool=name):
                parameters = definitions[name]["parameters"]
                self.assertEqual(parameters["required"], [field])
                self.assertEqual(set(parameters["properties"]), {field})
                self.assertFalse(parameters["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
