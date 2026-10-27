"""Observation tests: real shell commands in a temp folder, model mocked at HTTP level."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from memory import MemoryContext, open_memory, thread_config
from observation import (
    ObservedSandbox,
    Tracer,
    classify_output,
    create_observed_coder_graph,
    create_observed_orchestrator,
    failure_streak,
    observe,
    read_trace,
    summarise_output,
    summarise_traces,
    uninstall,
)
from observation.observation import NUDGE_AFTER, NUDGE_NOTE, STOP_AFTER, _add_error_detail
from agent_roles.executor import MAX_AGENT_ROUNDS
from scripts.tools.tool_registry import execute_tool

WORKSPACE = "/home/user/project"
PYTEST_FAIL = """\
FAILED tests/test_app.py::test_total - assert 3 == 4
========================= 1 failed, 2 passed in 0.12s ==========================
"""
TRACEBACK = """\
Traceback (most recent call last):
  File "app.py", line 3, in <module>
ModuleNotFoundError: No module named 'flask'
"""


class LocalSandbox:
    """Same _execute contract as the Minikube sandbox, run in a temp folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.save_to_host = Mock()

    def _execute(self, command, timeout=60, input_data=None):
        command = [part.replace(WORKSPACE, str(self.root)) for part in command]
        done = subprocess.run(command, input=input_data, capture_output=True, timeout=timeout)
        return SimpleNamespace(stdout=done.stdout, stderr=done.stderr, exit_code=done.returncode)


def call(name, arguments, call_id):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def tools(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def text(content):
    return {"role": "assistant", "content": content}


def review(status, feedback="checked"):
    return text(json.dumps({"status": status, "feedback": feedback, "summary": f"Review {status}"}))


def worker_done(status="complete", summary="Done."):
    return text(json.dumps({"status": status, "summary": summary}))


class FakeModel:
    """Answers OpenRouter POSTs in order and records what the agents were sent."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.requests = []

    def __call__(self, url, json=None, **kwargs):
        self.requests.append(json)
        message = self.replies.pop(0)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }
        return response


def plan(*tasks):
    content = json.dumps({"summary": "Plan", "tasks": [
        {"id": task_id, "title": title, "description": title, "depends_on": [], "files": []}
        for task_id, title in tasks]})
    return AIMessage(content=content, usage_metadata={"input_tokens": 50, "output_tokens": 5, "total_tokens": 55})


class ClassifyTest(unittest.TestCase):
    def test_failures_are_named(self) -> None:
        cases = [("", 0, False, None), (PYTEST_FAIL, 1, False, "test_fail"),
                 (TRACEBACK, 1, False, "import"), ("SyntaxError: bad", 1, False, "syntax"),
                 ("sh: 1: x: not found", 127, False, "command_not_found"),
                 ("cat: x: No such file or directory", 1, False, "not_found"),
                 ("", 124, True, "timeout"), ("odd", 2, False, "nonzero_exit")]
        for output, code, timed_out, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_output(output, code, timed_out), expected)

    def test_summaries(self) -> None:
        self.assertEqual(summarise_output(PYTEST_FAIL), "1 failed, 2 passed in 0.12s")
        self.assertEqual(summarise_output(TRACEBACK), "ModuleNotFoundError: No module named 'flask'")
        bad = observe(call("nope", {}, "1"), json.dumps({"error": "execute_tool: execute_tool: unknown tool"}))
        self.assertEqual((bad.ok, bad.error_type), (False, "bad_call"))

    def test_streak(self) -> None:
        fail = {"tool": "run_command", "arguments": {"command": "x"}, "ok": False,
                "error_type": "import", "summary": "s"}
        self.assertEqual(failure_streak([fail] * 3).count, 3)
        self.assertEqual(failure_streak([fail] * 3 + [{"ok": True}]).count, 0)


class ObservedSandboxTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.sandbox = ObservedSandbox(LocalSandbox(self.root), limit=1000)

    def test_failed_read_gets_its_reason(self) -> None:
        raw = execute_tool(call("read_file", {"path": "missing.py"}, "1"), sandbox=self.sandbox)
        self.assertIn("No such file or directory", json.loads(_add_error_detail(raw, self.sandbox))["error"])

    def test_binary_and_long_output_keep_the_end(self) -> None:
        (self.root / "a.bin").write_bytes(b"\xff" + b"x" * 50_000 + b"END")
        content = json.loads(execute_tool(call("read_file", {"path": "a.bin"}, "1"), sandbox=self.sandbox))["content"]
        self.assertTrue(content.endswith("END"))
        self.assertIn("characters cut", content)

    def test_list_files_json_is_left_whole(self) -> None:
        for index in range(300):
            (self.root / f"file_{index:03}.txt").write_text("x")
        result = json.loads(execute_tool(call("list_files", {"path": "."}, "1"), sandbox=self.sandbox))
        self.assertEqual(len(result["entries"]), 200)

    def test_host_sync_is_skipped_after_reads_only(self) -> None:
        (self.root / "a.txt").write_text("a")
        self.sandbox.save_to_host("p")                                   # first sync always runs
        execute_tool(call("read_file", {"path": "a.txt"}, "1"), sandbox=self.sandbox)
        self.sandbox.save_to_host("p")                                   # only a read: skipped
        execute_tool(call("write_file", {"path": "b.txt", "content": "b"}, "2"), sandbox=self.sandbox)
        self.sandbox.save_to_host("p")                                   # a write: runs
        self.assertEqual((self.sandbox.syncs, self.sandbox.skipped_syncs), (2, 1))


class GraphTestBase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(uninstall)
        self.root = Path(directory.name)
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (self.project / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\n"
            "class T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 2), 4)\n")
        self.tracer = Tracer(self.root / "traces")
        self.context = MemoryContext(project_id="project-a")
        self.sandbox = LocalSandbox(self.project)
        env = patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def run_orchestrator(self, plan_message, replies):
        model = FakeModel(replies)
        with open_memory(self.root / "memory.db") as (checkpointer, store):
            graph = create_observed_orchestrator(checkpointer=checkpointer, store=store, sandbox=self.sandbox,
                                                 project=self.project, tracer=self.tracer)
            with patch("requests.post", side_effect=model), \
                 patch("langchain_openai.ChatOpenAI.invoke", return_value=plan_message):
                result = graph.invoke({"messages": [{"role": "user", "content": "Fix add"}]},
                                      thread_config(self.context, "t1"), context=self.context)
        return result, model

    def events(self):
        files = list((self.root / "traces").glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        return read_trace(files[0])

    def report(self):
        return summarise_traces((self.root / "traces").glob("*.jsonl"))


FIX = {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}


def tests_call(call_id):
    return call("run_command", {"command": "PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -q"}, call_id)


class OrchestratorTest(GraphTestBase):
    def test_worker_and_reviewer_are_observed_and_traced(self) -> None:
        result, model = self.run_orchestrator(plan(("task-1", "Fix add")), [
            tools(tests_call("w1")),
            tools(call("write_file", FIX, "w2")),
            tools(tests_call("w3")),
            worker_done(summary="Fixed add; tests pass."),
            tools(tests_call("r1")),
            review("PASS"),
        ])
        self.assertEqual(result["review"]["status"], "PASS")

        # The worker saw the structured observation of its failing test run.
        tool_message = next(m for m in model.requests[1]["messages"] if m["role"] == "tool")
        seen = json.loads(tool_message["content"])
        self.assertEqual((seen["ok"], seen["error_type"]), (False, "test_fail"))
        self.assertIn("FAILED", seen["summary"])

        events = self.events()
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[0], "turn_start")
        self.assertEqual(kinds[-1], "turn_end")
        self.assertEqual(events[-1]["status"], "completed")
        self.assertIn("plan", kinds)
        agents = {(e["agent"], e.get("task_id")) for e in events if e["event"] == "tool"}
        self.assertEqual(agents, {("worker", "task-1"), ("reviewer", None)})

        report = self.report()
        self.assertEqual(report["success_rate"], 1.0)
        self.assertEqual(report["first_review_pass_rate"], 1.0)
        self.assertEqual(report["recovery_rate"], 1.0)
        # 1 planner call (55) + 6 coder calls (110 each).
        self.assertEqual(report["tokens"]["total"], 55 + 6 * 110)
        self.assertEqual(set(report["by_agent"]), {"planner", "worker", "reviewer"})

    def test_stuck_worker_becomes_a_failed_task_and_is_repaired(self) -> None:
        stuck = [tools(call("run_command", {"command": "python3 -c 'import flask_missing'"}, f"s{i}"))
                 for i in range(STOP_AFTER)]
        result, model = self.run_orchestrator(plan(("task-1", "Fix add")), stuck + [
            # Worker loop already ended by the observer; the reviewer fails it.
            review("FAIL", "add still subtracts"),
            # Rework worker fixes it, reviewer passes.
            tools(call("write_file", FIX, "f1")),
            worker_done(),
            review("PASS"),
        ])
        self.assertEqual(result["review"]["status"], "PASS")
        # Exactly STOP_AFTER worker model calls: the observer answered the next one itself.
        worker_requests = model.requests[:STOP_AFTER]
        notes = [m for request in worker_requests for m in request["messages"]
                 if m["role"] == "tool" and NUDGE_NOTE in m["content"]]
        self.assertEqual(len(notes), STOP_AFTER - NUDGE_AFTER)  # nudge stays in later history
        failed = result["work_history"][0]
        self.assertEqual(failed["status"], "failed")
        self.assertIn("Stopped by observer", failed["summary"])

        kinds = [e["event"] for e in self.events()]
        self.assertEqual(kinds.count("stuck_nudge"), 1)
        self.assertEqual(kinds.count("stuck_stop"), 1)
        report = self.report()
        self.assertEqual((report["worker_tasks"], report["worker_tasks_failed"]), (2, 1))
        self.assertEqual(report["mean_reviews"], 2)
        self.assertEqual(report["task_recovery_rate"], 1.0)

    def test_round_limit_becomes_a_failed_task(self) -> None:
        echoes = [
            tools(call("run_command", {"command": f"echo {i}"}, f"e{i}"))
            for i in range(MAX_AGENT_ROUNDS + 1)
        ]
        result, _ = self.run_orchestrator(plan(("task-1", "Loop")), echoes + [
            review("FAIL", "nothing done"), tools(call("write_file", FIX, "f1")), worker_done(), review("PASS"),
        ])
        self.assertEqual(result["work_history"][0]["status"], "failed")
        self.assertIn("round tool limit", result["work_history"][0]["summary"])

    def test_stuck_reviewer_becomes_a_fail_review(self) -> None:
        stuck = [tools(call("read_file", {"path": "nope.py"}, f"r{i}")) for i in range(STOP_AFTER)]
        result, _ = self.run_orchestrator(plan(("task-1", "Fix add")), [
            tools(call("write_file", FIX, "w1")), worker_done(),
        ] + stuck + [
            tools(call("write_file", FIX, "w2")), worker_done(), review("PASS"),
        ])
        events = self.events()
        reviews = [e for e in events if e["event"] == "review"]
        self.assertEqual([r["status"] for r in reviews], ["FAIL", "PASS"])
        self.assertIn("Stopped by observer", reviews[0]["feedback"])
        self.assertEqual(result["review"]["status"], "PASS")

    def test_no_task_plan_is_answered(self) -> None:
        answer = AIMessage(content=json.dumps({"summary": "It adds two numbers.", "tasks": []}))
        result, _ = self.run_orchestrator(answer, [])
        self.assertEqual(result["messages"][-1]["content"], "It adds two numbers.")
        self.assertEqual(self.events()[-1]["status"], "answered")

    def test_existing_chat_loop_runs_it(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        from main import chat
        model = FakeModel([tools(call("write_file", FIX, "w1")), worker_done(), review("PASS")])
        output = StringIO()
        with open_memory(self.root / "memory.db") as (checkpointer, store):
            graph = create_observed_orchestrator(checkpointer=checkpointer, store=store, sandbox=self.sandbox,
                                                 project=self.project, tracer=self.tracer)
            with patch("builtins.input", side_effect=["Fix add", "exit"]), \
                 patch("requests.post", side_effect=model), \
                 patch("langchain_openai.ChatOpenAI.invoke", return_value=plan(("task-1", "Fix add"))), \
                 redirect_stdout(output):
                chat(graph, store, self.context, "t1")
        self.assertIn("Review PASS", output.getvalue())
        self.assertEqual(self.events()[-1]["status"], "completed")


class BaselineTest(GraphTestBase):
    def test_baseline_coder_is_observed_and_stopped(self) -> None:
        model = FakeModel([tools(call("run_command", {"command": "python3 -c 'import nope_x'"}, f"b{i}"))
                           for i in range(STOP_AFTER)])
        with open_memory(self.root / "memory.db") as (checkpointer, store):
            graph = create_observed_coder_graph(checkpointer=checkpointer, store=store, sandbox=self.sandbox,
                                                project=self.project, tracer=self.tracer)
            with patch("requests.post", side_effect=model):
                result = graph.invoke({"messages": [{"role": "user", "content": "Run it"}]},
                                      thread_config(self.context, "b1"), context=self.context)
        self.assertTrue(result["messages"][-1]["content"].startswith("Stopped by observer"))
        self.assertEqual(self.events()[-1]["status"], "stopped")
        self.assertEqual(set(self.report()["by_agent"]), {"coder"})


class UninstallTest(unittest.TestCase):
    def test_uninstall_restores_original_functions(self) -> None:
        import agent_roles.coder as coder
        import agent_roles.graph as graph
        import requests
        before = graph.execute_task
        create_observed_orchestrator(checkpointer=None, store=None, sandbox=Mock(), project=Path("."))
        self.assertIsNot(graph.execute_task, before)
        uninstall()
        self.assertIs(graph.execute_task, before)
        self.assertIs(coder.requests, requests)


if __name__ == "__main__":
    unittest.main()
