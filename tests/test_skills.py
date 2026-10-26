"""Skill discovery, safe resource reads, and integration with the actual agent loops."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from agent_roles.graph import create_coder_graph
from main import chat
from memory import MemoryContext, open_memory, thread_config
from observation import Tracer, create_observed_orchestrator, read_trace, uninstall
from scripts.tools.tool_registry import execute_tool, list_tools
from skill_library import MAX_FILE_BYTES, SkillLibrary


BODY = "BODY_ONLY_SENTINEL: inspect the existing interface before changing its layout."
RESOURCE = "RESOURCE_ONLY_SENTINEL: check the narrow viewport and keyboard navigation."
SKILL_TOOLS = {"list_skills", "read_skill", "read_skill_resource"}


def write_skill(root, name="ui-kit", *, description="Guidance for interface work.", body=BODY):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {json.dumps(description)}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return folder


def tool_call(name, arguments, call_id):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def tool_reply(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def answer(content):
    return {"role": "assistant", "content": content}


class ScriptedHTTP:
    """Keep the real coder adapter and tool loop; replace only the external model."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def __call__(self, url, *, json, **kwargs):
        self.requests.append(deepcopy(json))
        message = next(self.replies)
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return response


class SkillLibraryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bundled = self.root / "bundled"
        self.project = self.root / "project"
        self.project.mkdir()
        self.folder = write_skill(self.bundled)
        (self.folder / "references").mkdir()
        (self.folder / "references" / "checks.md").write_text(RESOURCE, encoding="utf-8")
        self.library = SkillLibrary(self.project, bundled=self.bundled)

    def test_catalog_has_metadata_only_and_resources_are_loaded_on_demand(self):
        expected = [{"name": "ui-kit", "description": "Guidance for interface work.", "source": "bundled"}]
        self.assertEqual(self.library.list_skills(), {"skills": expected, "errors": []})
        prompt = self.library.catalog_prompt()
        self.assertIn("ui-kit", prompt)
        self.assertNotIn(BODY, prompt)
        self.assertNotIn(RESOURCE, prompt)
        skill = self.library.read_skill("ui-kit")
        self.assertIn(BODY, skill["content"])
        self.assertEqual(skill["resources"], ["references/checks.md"])
        self.assertEqual(self.library.read_skill_resource("ui-kit", "references/checks.md"), {
            "name": "ui-kit", "path": "references/checks.md", "content": RESOURCE,
        })

    def test_folded_yaml_and_new_skills_are_available_without_rebuilding_library(self):
        added = self.bundled / "layout-review"
        added.mkdir()
        (added / "SKILL.md").write_text(
            "---\nname: layout-review\ndescription: >-\n"
            "  Inspect layouts across\n  small and large screens.\n---\nCheck the layout.\n",
            encoding="utf-8",
        )
        catalog = {item["name"]: item for item in self.library.list_skills()["skills"]}
        self.assertEqual(catalog["layout-review"]["description"],
                         "Inspect layouts across small and large screens.")
        self.assertIn("Check the layout.", self.library.read_skill("layout-review")["content"])
        (added / "SKILL.md").unlink()
        with self.assertRaises(ValueError):
            self.library.read_skill("layout-review")

    def test_project_override_is_local_and_same_root_is_not_duplicated(self):
        project_folder = write_skill(self.project / "skills", body="PROJECT_BODY_SENTINEL")
        (project_folder / "local.md").write_text("project resource", encoding="utf-8")
        other_project = self.root / "other-project"
        other_project.mkdir()
        other = SkillLibrary(other_project, bundled=self.bundled)
        overridden = self.library.read_skill("ui-kit")
        self.assertEqual(overridden["source"], "project")
        self.assertIn("PROJECT_BODY_SENTINEL", overridden["content"])
        self.assertEqual(overridden["resources"], ["local.md"])
        self.assertEqual(other.read_skill("ui-kit")["source"], "bundled")
        self.assertNotIn("PROJECT_BODY_SENTINEL", other.read_skill("ui-kit")["content"])
        with self.assertRaises(ValueError):
            other.read_skill_resource("ui-kit", "local.md")
        same_root = SkillLibrary(self.project, bundled=self.project / "skills")
        self.assertEqual(len(same_root.list_skills()["skills"]), 1)
        self.assertEqual(same_root.list_skills()["errors"], [])

    def test_malformed_skill_is_isolated_and_invalid_override_keeps_valid_bundle(self):
        malformed = {
            "missing-header": BODY,
            "unclosed-header": "---\nname: unclosed-header\ndescription: Test\n",
            "wrong-name": "---\nname: another-name\ndescription: Test\n---\nBody",
            "bad-description": "---\nname: bad-description\ndescription: []\n---\nBody",
            "empty-body": "---\nname: empty-body\ndescription: Test\n---\n",
            "bad-yaml": "---\nname: bad-yaml\ndescription: [\n---\nBody",
        }
        for name, content in malformed.items():
            folder = self.bundled / name
            folder.mkdir()
            (folder / "SKILL.md").write_text(content, encoding="utf-8")
        invalid_override = write_skill(self.project / "skills")
        (invalid_override / "SKILL.md").write_bytes(b"\xff\xfeinvalid UTF-8")
        catalog = self.library.list_skills()
        self.assertEqual([item["name"] for item in catalog["skills"]], ["ui-kit"])
        self.assertEqual(catalog["skills"][0]["source"], "bundled")
        self.assertEqual({item["name"] for item in catalog["errors"]}, {*malformed, "ui-kit"})
        self.assertIn(BODY, self.library.read_skill("ui-kit")["content"])

    def test_oversized_and_binary_files_fail_without_truncating_instructions(self):
        oversized = write_skill(self.bundled, "oversized", body="x" * MAX_FILE_BYTES)
        catalog = self.library.list_skills()
        self.assertEqual([item["name"] for item in catalog["skills"]], ["ui-kit"])
        self.assertEqual(catalog["errors"][0]["name"], oversized.name)
        with self.assertRaises(ValueError):
            self.library.read_skill("oversized")
        cases = {"too-big.txt": b"x" * (MAX_FILE_BYTES + 1), "binary.txt": b"hello\0world",
                 "invalid.txt": b"\xff"}
        for name, data in cases.items():
            with self.subTest(name=name):
                (self.folder / name).write_bytes(data)
                with self.assertRaises(ValueError):
                    self.library.read_skill_resource("ui-kit", name)
        (self.folder / "limit.txt").write_bytes(b"x" * MAX_FILE_BYTES)
        self.assertEqual(len(self.library.read_skill_resource("ui-kit", "limit.txt")["content"]),
                         MAX_FILE_BYTES)

    def test_resource_paths_cannot_escape_the_skill_or_follow_symlinks(self):
        outside = self.root / "private.txt"
        outside.write_text("OUTSIDE_SENTINEL", encoding="utf-8")
        (self.folder / "outside.txt").symlink_to(outside)
        (self.folder / "local-link.md").symlink_to(self.folder / "references" / "checks.md")
        (self.folder / "linked-dir").symlink_to(self.root, target_is_directory=True)
        for path in ("", ".", "..", "../private.txt", str(outside),
                     "C:/private.txt", "references\\checks.md", "outside.txt",
                     "local-link.md", "linked-dir/private.txt", "references", "missing.md"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.library.read_skill_resource("ui-kit", path)
        self.assertEqual(self.library.read_skill("ui-kit")["resources"], ["references/checks.md"])
        with self.assertRaises(ValueError):
            self.library.read_skill("../ui-kit")

    def test_symlinked_skill_file_folder_and_root_are_not_discovered(self):
        external = write_skill(self.root / "external", "linked-folder")
        (self.bundled / "linked-folder").symlink_to(external, target_is_directory=True)
        linked_file = self.bundled / "linked-file"
        linked_file.mkdir()
        (linked_file / "SKILL.md").symlink_to(self.folder / "SKILL.md")
        catalog = self.library.list_skills()
        self.assertEqual([item["name"] for item in catalog["skills"]], ["ui-kit"])
        self.assertEqual({item["name"] for item in catalog["errors"]}, {"linked-file", "linked-folder"})
        linked_root = self.root / "linked-root"
        linked_root.symlink_to(self.bundled, target_is_directory=True)
        root_catalog = SkillLibrary(bundled=linked_root).list_skills()
        self.assertEqual(root_catalog["skills"], [])
        self.assertTrue(root_catalog["errors"])

    def test_tool_arguments_cannot_override_library_scope(self):
        sandbox = Mock()
        malformed = [
            ("list_skills", {"project": str(self.root)}),
            ("read_skill", {"name": "ui-kit", "root": str(self.root)}),
            ("read_skill", {"name": 123}),
            ("read_skill_resource", {"name": "ui-kit"}),
            ("read_skill_resource", {"name": "ui-kit", "path": "../private.txt"}),
        ]
        for index, (name, arguments) in enumerate(malformed):
            with self.subTest(name=name, arguments=arguments):
                result = json.loads(execute_tool(tool_call(name, arguments, str(index)),
                                                 sandbox=sandbox, skills=self.library))
                self.assertIn("error", result)
        definitions = {item["function"]["name"]: item["function"]["parameters"] for item in list_tools()}
        for name in SKILL_TOOLS:
            self.assertFalse(definitions[name]["additionalProperties"])
        self.assertEqual(set(definitions["read_skill_resource"]["properties"]), {"name", "path"})
        sandbox._execute.assert_not_called()


class SkillRuntimeTest(unittest.TestCase):
    def setUp(self):
        uninstall()
        self.addCleanup(uninstall)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.project = self.root / "project"
        self.project.mkdir()
        bundled = self.root / "bundled"
        folder = write_skill(bundled)
        (folder / "checks.md").write_text(RESOURCE, encoding="utf-8")
        self.library = SkillLibrary(self.project, bundled=bundled)
        self.sandbox = Mock()
        self.context = MemoryContext(project_id="skills-project", user_id="alice")
        environment = patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only-key"})
        environment.start()
        self.addCleanup(environment.stop)

    def assert_catalog_and_tools(self, payload):
        systems = [item for item in payload["messages"] if item["role"] == "system"]
        self.assertEqual(len(systems), 1)
        self.assertIn("ui-kit", systems[0]["content"])
        self.assertIn("Guidance for interface work.", systems[0]["content"])
        self.assertNotIn(BODY, systems[0]["content"])
        self.assertNotIn(RESOURCE, systems[0]["content"])
        names = {item["function"]["name"] for item in payload["tools"]}
        self.assertTrue(SKILL_TOOLS.issubset(names))

    def test_single_agent_reads_skill_and_resource_through_real_tool_loop(self):
        model = ScriptedHTTP([
            tool_reply(tool_call("list_skills", {}, "catalog")),
            tool_reply(tool_call("read_skill", {"name": "ui-kit"}, "instructions")),
            tool_reply(tool_call("read_skill_resource", {"name": "ui-kit", "path": "checks.md"}, "resource")),
            answer("Applied the interface guidance."),
        ])
        with open_memory(self.root / "memory.db") as (checkpointer, store):
            graph = create_coder_graph(checkpointer=checkpointer, store=store, sandbox=self.sandbox,
                                       project=self.project, skills=self.library)
            with patch("requests.post", side_effect=model):
                result = graph.invoke({"messages": [{"role": "user", "content": "Use $ui-kit."}]},
                                      thread_config(self.context, "single"), context=self.context)
        self.assertEqual(len(model.requests), 4)
        for payload in model.requests:
            self.assert_catalog_and_tools(payload)
        self.assertNotIn(BODY, json.dumps(model.requests[0]))
        self.assertNotIn(BODY, json.dumps(model.requests[1]))
        self.assertIn(BODY, json.loads(model.requests[2]["messages"][-1]["content"])["content"])
        self.assertEqual(json.loads(model.requests[3]["messages"][-1]["content"])["content"], RESOURCE)
        outputs = [item for item in result["messages"] if item["role"] == "tool"]
        self.assertEqual([item["tool_call_id"] for item in outputs], ["catalog", "instructions", "resource"])
        self.assertEqual(result["messages"][-1]["content"], "Applied the interface guidance.")
        self.sandbox._execute.assert_not_called()
        self.sandbox.save_to_host.assert_not_called()

    def test_observed_workflow_gives_catalog_to_planner_and_skill_tools_to_both_roles(self):
        plan = AIMessage(content=json.dumps({"summary": "Use the interface skill.", "tasks": [{
            "id": "task-1", "title": "Inspect layout", "description": "Read ui-kit and inspect the layout.",
            "depends_on": [], "files": [],
        }]}))
        model = ScriptedHTTP([
            tool_reply(tool_call("read_skill", {"name": "ui-kit"}, "worker-skill")),
            tool_reply(tool_call("read_skill_resource", {"name": "ui-kit", "path": "checks.md"}, "worker-resource")),
            answer("Inspected the interface."),
            tool_reply(tool_call("list_skills", {}, "reviewer-catalog"),
                       tool_call("read_skill", {"name": "ui-kit"}, "reviewer-skill"),
                       tool_call("read_skill_resource", {"name": "ui-kit", "path": "checks.md"}, "reviewer-resource")),
            answer(json.dumps({"status": "PASS", "feedback": "Guidance checked.", "summary": "Verified."})),
        ])
        trace_dir = self.root / "traces"
        with open_memory(self.root / "memory.db") as (checkpointer, store):
            graph = create_observed_orchestrator(
                checkpointer=checkpointer, store=store, sandbox=self.sandbox, project=self.project,
                skills=self.library, tracer=Tracer(trace_dir),
            )
            with patch("requests.post", side_effect=model), \
                 patch("langchain_openai.ChatOpenAI.invoke", return_value=plan) as planner_model:
                result = graph.invoke({"messages": [{"role": "user", "content": "Use $ui-kit."}]},
                                      thread_config(self.context, "workflow"), context=self.context)
        planner_messages = planner_model.call_args.args[-1]
        self.assertIn("ui-kit", planner_messages[0]["content"])
        self.assertNotIn(BODY, json.dumps(planner_messages))
        self.assertEqual(result["review"]["status"], "PASS")
        self.assertEqual(len(model.requests), 5)
        for payload in model.requests:
            self.assert_catalog_and_tools(payload)
        worker_contract = model.requests[0]["messages"][-1]
        worker_payload = json.loads(worker_contract["content"])
        self.assertEqual(worker_payload["task"]["id"], "task-1")
        self.assertEqual(worker_payload["user_request"], "Use $ui-kit.")
        worker_user_messages = [item for item in model.requests[0]["messages"] if item["role"] == "user"]
        for payload in model.requests[1:3]:
            self.assertEqual([item for item in payload["messages"] if item["role"] == "user"],
                             worker_user_messages)
        self.assertIn(BODY, json.loads(model.requests[1]["messages"][-1]["content"])["content"])
        self.assertEqual(json.loads(model.requests[2]["messages"][-1]["content"])["content"], RESOURCE)
        self.assertNotIn(BODY, json.dumps(model.requests[3]))
        reviewer_contract = model.requests[3]["messages"][-1]
        reviewer_payload = json.loads(reviewer_contract["content"])
        self.assertEqual(reviewer_payload["original_request"], "Use $ui-kit.")
        self.assertEqual(reviewer_payload["planned_tasks"][0]["id"], "task-1")
        self.assertEqual(reviewer_payload["worker_results_including_repairs"][0]["summary"],
                         "Inspected the interface.")
        self.assertEqual([item for item in model.requests[4]["messages"] if item["role"] == "user"],
                         [item for item in model.requests[3]["messages"] if item["role"] == "user"])
        reviewer_outputs = [json.loads(item["content"]) for item in model.requests[4]["messages"]
                            if item["role"] == "tool"]
        self.assertEqual(len(reviewer_outputs), 3)
        self.assertIn(BODY, reviewer_outputs[1]["content"])
        self.assertEqual(reviewer_outputs[2]["content"], RESOURCE)
        reviewer_tools = {item["function"]["name"] for item in model.requests[3]["tools"]}
        self.assertNotIn("write_file", reviewer_tools)
        events = [event for path in trace_dir.glob("*.jsonl") for event in read_trace(path)
                  if event["event"] == "tool"]
        self.assertEqual(len(events), 5)
        self.assertEqual({item["agent"] for item in events}, {"worker", "reviewer"})
        self.assertTrue(all(item["ok"] for item in events))
        self.sandbox._execute.assert_not_called()
        self.sandbox.save_to_host.assert_not_called()

    def test_cli_inspection_and_bad_names_do_not_call_model_or_create_history(self):
        commands = ["/skills", "/skill ui-kit", "/skill missing", "/skill", "exit"]
        output = StringIO()
        with open_memory(self.root / "memory.db") as (checkpointer, store):
            graph = create_coder_graph(checkpointer=checkpointer, store=store, sandbox=self.sandbox,
                                       project=self.project, skills=self.library)
            with patch("builtins.input", side_effect=commands), patch("requests.post") as model, \
                 redirect_stdout(output):
                chat(graph, store, self.context, "cli", skills=self.library)
            model.assert_not_called()
            self.assertFalse(graph.get_state(thread_config(self.context, "cli")).values)
        self.assertIn("ui-kit", output.getvalue())
        self.assertIn(BODY, output.getvalue())
        self.assertIn("checks.md", output.getvalue())
        self.sandbox._execute.assert_not_called()
        self.sandbox.save_to_host.assert_not_called()


if __name__ == "__main__":
    unittest.main()
