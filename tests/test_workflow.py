"""Worker contracts and LangGraph recovery without API or sandbox services."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_roles.executor import AgentExecutionError, execute_task, run_agent
from agent_roles.graph import create_orchestrator_graph
from agent_roles.reviewer import review_work
from memory import MemoryContext, open_memory, thread_config


def contract(task_id="task-1", dependencies=None):
    return {
        "id": task_id, "title": f"Implement {task_id}", "description": "Fix the sum function",
        "depends_on": dependencies or [], "files": ["sum.py"],
    }


def reply(text="", calls=None):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
        message["reasoning_details"] = [{"type": "reasoning.encrypted", "data": "opaque"}]
    return {"text": text, "message": message, "tool_calls": calls or []}


def final_worker(status="complete", summary="Fixed and checked."):
    return reply(json.dumps({"status": status, "summary": summary}))


def read_call():
    return {"id": "read-1", "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": "sum.py"}),
    }}


class WorkerTest(unittest.TestCase):
    def test_worker_and_reviewer_keep_assignment_after_each_tool_round(self):
        for role in ("worker", "reviewer"):
            with self.subTest(role=role):
                history = [{"role": "user", "content": "Original request"}]
                observed = []
                tool_reply = reply(calls=[read_call()])

                def model(text, **kwargs):
                    observed.append(deepcopy(kwargs["context"]))
                    self.assertIsNone(text)  # Assignment already lives in context.
                    if len(observed) < 3:
                        return deepcopy(tool_reply)
                    return final_worker() if role == "worker" else reply(json.dumps({
                        "status": "PASS", "feedback": "Checked", "summary": "Done",
                    }))

                with patch("agent_roles.executor.send_to_coder", side_effect=model):
                    if role == "worker":
                        execute_task(contract(), request="Fix sum", history=history,
                                     previous_results=[], review_feedback="Keep integers",
                                     run_tool=lambda call: "file contents")
                    else:
                        review_work("Fix sum", [contract()], [], history=history,
                                    run_tool=lambda call: "file contents")

                self.assertEqual(len(observed), 3)
                assignment = observed[0][-1]
                for context in observed:
                    self.assertEqual(context[1], assignment)
                    self.assertEqual(context.count(assignment), 1)
                self.assertIn("Fix sum", assignment["content"])
                if role == "worker":
                    self.assertIn("Keep integers", assignment["content"])
                else:
                    self.assertIn("planned_tasks", assignment["content"])
                self.assertEqual(observed[1][-2], tool_reply["message"])
                self.assertEqual(observed[1][-1]["tool_call_id"], "read-1")
                self.assertEqual(history, [{"role": "user", "content": "Original request"}])

    def test_worker_preserves_reported_status(self):
        for status in ("complete", "incomplete", "failed"):
            with self.subTest(status=status):
                with patch("agent_roles.executor.send_to_coder", return_value=final_worker(status)):
                    result = execute_task(contract(), request="Fix", history=[],
                                          previous_results=[], run_tool=Mock())
                self.assertEqual(result["status"], status)

    def test_invalid_or_empty_worker_reports_are_incomplete(self):
        for text in ("Cannot finish", "", "[]", "null", '{"status":"complete"}',
                     '{"status":[],"summary":"bad"}',
                     '{"status":"complete","summary":" "}'):
            with self.subTest(text=text):
                with patch("agent_roles.executor.send_to_coder", return_value=reply(text)):
                    result = execute_task(contract(), request="Fix", history=[],
                                          previous_results=[], run_tool=Mock())
                self.assertEqual(result["status"], "incomplete")

    def test_model_failure_becomes_failed_task(self):
        with patch("agent_roles.executor.send_to_coder", side_effect=RuntimeError("API unavailable")):
            result = execute_task(contract(), request="Fix", history=[],
                                  previous_results=[], run_tool=Mock())
        self.assertEqual(result["status"], "failed")
        self.assertIn("API unavailable", result["summary"])

    def test_invalid_reviewer_status_raises_report_error(self):
        for status in ([], {}, None, "UNKNOWN"):
            with self.subTest(status=status):
                with patch("agent_roles.executor.send_to_coder", return_value=reply(json.dumps({
                    "status": status, "feedback": "Checked", "summary": "Done",
                }))):
                    with self.assertRaises(ValueError):
                        review_work("Fix sum", [contract()], [], history=[], run_tool=Mock())

    def test_final_answer_after_last_allowed_tool_round_is_accepted(self):
        tool = Mock(return_value="contents")
        with patch("agent_roles.executor.send_to_coder", side_effect=[
            reply(calls=[read_call()]), reply("Done"),
        ]):
            result = run_agent("Fix", history=[], system_prompt="worker", model=None,
                               run_tool=tool, max_rounds=1)
        self.assertEqual(result, {"text": "Done", "tool_count": 1})
        tool.assert_called_once()

    def test_action_budget_stops_repeated_tool_calls(self):
        tool = Mock(return_value="contents")
        with patch("agent_roles.executor.send_to_coder", return_value=reply(calls=[read_call()])):
            with self.assertRaises(AgentExecutionError):
                run_agent("Fix", history=[], system_prompt="worker", model=None,
                          run_tool=tool, max_rounds=1)
        tool.assert_called_once()

    def test_infrastructure_failure_stays_retryable(self):
        # Do not turn a host-sync failure into an ordinary failed task: preserve
        # the pending LangGraph node and its completed tool cache for /retry.
        with patch("agent_roles.executor.send_to_coder", return_value=reply(calls=[read_call()])):
            with self.assertRaisesRegex(RuntimeError, "host sync failed"):
                execute_task(contract(), request="Fix", history=[], previous_results=[],
                             run_tool=Mock(side_effect=RuntimeError("host sync failed")))


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.context = MemoryContext(project_id="workflow-test")
        self.config = thread_config(self.context, "one")

    def invoke(self, tasks, model, review, max_rework_attempts=2):
        with open_memory(self.project / "memory.sqlite") as (checkpointer, store):
            graph = create_orchestrator_graph(checkpointer=checkpointer, store=store,
                                             sandbox=Mock(), project=self.project,
                                             max_rework_attempts=max_rework_attempts)
            with patch("agent_roles.graph.create_plan", return_value={"summary": "Plan", "tasks": tasks}):
                with patch("agent_roles.executor.send_to_coder", side_effect=model) as worker:
                    with patch("agent_roles.graph.review_work", side_effect=review) as reviewer:
                        result = graph.invoke({"messages": [{"role": "user", "content": "Fix sum"}]},
                                              self.config, context=self.context)
            self.assertFalse(graph.get_state(self.config).next)
        return result, worker, reviewer

    def test_failed_dependency_is_skipped_and_repair_recovers(self):
        tasks = [contract(), contract("task-2", ["task-1"]), contract("task-3")]
        result, worker, reviewer = self.invoke(
            tasks,
            [final_worker("incomplete", "Missing implementation"), final_worker(), final_worker()],
            [{"status": "PASS", "feedback": "Overlooked failure", "summary": "Done"},
             {"status": "PASS", "feedback": "All tasks verified", "summary": "Recovered"}],
        )
        self.assertEqual([item["status"] for item in result["work_history"]],
                         ["incomplete", "skipped", "complete", "complete"])
        self.assertEqual(worker.call_count, 3)  # Dependent task did not run.
        self.assertEqual(reviewer.call_count, 2)
        repair_prompt = json.loads(worker.call_args.kwargs["context"][-1]["content"])
        self.assertIn("task-2 (skipped)", repair_prompt["review_feedback_to_address"])
        self.assertEqual(result["messages"][-1]["content"], "Recovered")

    def test_worker_exception_enters_repair_without_manual_retry(self):
        result, worker, reviewer = self.invoke(
            [contract()], [RuntimeError("provider failure"), final_worker()],
            [{"status": "FAIL", "feedback": "Retry task", "summary": "Failed"},
             {"status": "PASS", "feedback": "Checked", "summary": "Recovered"}],
        )
        self.assertEqual(result["work_history"][0]["status"], "failed")
        self.assertEqual(worker.call_count, 2)
        self.assertEqual(result["messages"][-1]["content"], "Recovered")

    def test_reviewer_failure_is_bounded_and_cannot_report_success(self):
        for error in (AgentExecutionError("API unavailable"), ValueError("Invalid JSON")):
            with self.subTest(error=error):
                result, worker, reviewer = self.invoke(
                    [contract()], [final_worker(), final_worker()], [error, error],
                    max_rework_attempts=1,
                )
                self.assertEqual(worker.call_count, 2)
                self.assertEqual(reviewer.call_count, 2)
                self.assertEqual(result["review"]["status"], "FAIL")
                self.assertIn(str(error), result["messages"][-1]["content"])
                self.assertEqual(result["review_attempt"], 2)

    def test_incomplete_task_cannot_pass_even_if_reviewer_says_pass(self):
        result, _, _ = self.invoke(
            [contract()], [final_worker("failed", "Cannot finish")],
            [{"status": "PASS", "feedback": "Looks fine", "summary": "Success"}],
            max_rework_attempts=0,
        )
        self.assertEqual(result["review"]["status"], "FAIL")
        self.assertIn("Cannot finish", result["messages"][-1]["content"])

    def test_orchestrator_retry_does_not_replay_completed_tool_after_sync_failure(self):
        sandbox = Mock()
        sandbox._execute.return_value = SimpleNamespace(stdout=b"contents", stderr=b"", exit_code=0)
        sandbox.save_to_host.side_effect = [RuntimeError("host sync failed"), None]
        with open_memory(self.project / "memory.sqlite") as (checkpointer, store):
            graph = create_orchestrator_graph(checkpointer=checkpointer, store=store,
                                             sandbox=sandbox, project=self.project)
            with patch("agent_roles.graph.create_plan", return_value={
                "summary": "Fix", "tasks": [contract()],
            }):
                with patch("agent_roles.executor.send_to_coder", side_effect=[
                    reply(calls=[read_call()]), reply(calls=[read_call()]), final_worker(),
                ]):
                    with patch("agent_roles.graph.review_work", return_value={
                        "status": "PASS", "feedback": "Checked", "summary": "Recovered",
                    }):
                        with self.assertRaisesRegex(RuntimeError, "host sync failed"):
                            graph.invoke({"messages": [{"role": "user", "content": "Fix sum"}]},
                                         self.config, context=self.context)
                        self.assertTrue(graph.get_state(self.config).next)
                        result = graph.invoke(None, self.config, context=self.context)
            self.assertFalse(graph.get_state(self.config).next)
        sandbox._execute.assert_called_once()
        self.assertEqual(sandbox.save_to_host.call_count, 2)
        self.assertEqual(result["messages"][-1]["content"], "Recovered")


if __name__ == "__main__":
    unittest.main()
