"""Observation it keeps an eye on what the agents are doing.

Every time an agent runs a tool, this file will look at the result and it will work out
if it worked, what went wrong,etc.. 
It also notices when an agent keeps making the same mistake, stops it before it wastes
the whole run, and writes everything to a log so we can measure how well the
system works.

To see the numbers, run:  python -m observation

"""

from __future__ import annotations

import argparse
from contextvars import ContextVar
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from scripts.tools.terminal_use import OUTPUT_LIMIT
from scripts.tools.tool_registry import SPECS


# Settings if you want to change them


DEFAULT_TRACE_DIR = Path.home() / ".local/share/cos30018/traces"

# how many errors, before the agent is told to change approach
NUDGE_AFTER = 3
# if it still keeps going, we stop it here.
STOP_AFTER = 5

NUDGE_NOTE = "observer_note"   # the key our note is added under in the tool result
SUMMARY_LIMIT = 200            # longest summary we keep, in characters
_CUT_MARKER = "characters cut]"   # text we put where long output was trimmed

# The error message run_agent() gets when an agent uses up all its rounds.
_ROUND_LIMIT = re.compile(r"exceeded its \d+-round tool limit")


# Reading tool results
#
# did it work (ok), what kind of problem it was (error_type), and the (summary).


# How we recognise each kind of error. check them in this order and use the first one that matches, so the more specific ones go first.
_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("syntax", re.compile(r"\b(SyntaxError|IndentationError|TabError)\b")),
    ("import", re.compile(r"\b(ModuleNotFoundError|ImportError)\b")),
    ("command_not_found", re.compile(r"command not found|: not found\b")),
    ("test_fail", re.compile(r"^FAILED\b|^=+ .*\bfailed\b.* =+$", re.MULTILINE)),
    ("permission", re.compile(r"Permission denied", re.IGNORECASE)),
    ("not_found", re.compile(r"No such file or directory|FileNotFoundError|Not a directory", re.IGNORECASE)),
    ("exception", re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)),
)
# Lines we look for when writing the summary: pytest's "1 failed, 2 passed"
# line, unittest's "Ran 3 tests" + "OK"/"FAILED", or an exception like
# "ModuleNotFoundError: No module named 'flask'".
_PYTEST_SUMMARY = re.compile(
    r"^=+ (.*\b(?:passed|failed|error|errors|skipped|no tests ran)\b.*?) =+$", re.MULTILINE)
_UNITTEST_RAN = re.compile(r"^Ran \d+ tests? in [\d.]+s$", re.MULTILINE)
_UNITTEST_RESULT = re.compile(r"^(OK(?: \(.*\))?|FAILED \(.*\))$", re.MULTILINE)
_EXCEPTION_LINE = re.compile(
    r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))(?::.*)?$", re.MULTILINE)


@dataclass
class Observation:
    """What happened when an agent used one tool.

    ok          True if it worked.
    summary     The one line worth reading, e.g. "1 failed, 2 passed".
    error_type  What kind of problem it was (None if it worked).
    The rest (exit code, timing, arguments) is kept for the logs.
    """

    tool: str
    ok: bool
    summary: str
    error_type: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    truncated: bool = False
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Squash text onto one line and cut it down if it's too long.
def _short(text: str, limit: int = SUMMARY_LIMIT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# Tools return JSON text. Turn it back into Python, or treat it as an error, message if it isn't valid JSON.
def _parse(raw: Any) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {"error": str(raw)}


def classify_output(text: str, exit_code: int | None = None, timed_out: bool = False) -> str | None:
    """Look at a command's output and say what kind of failure it was.

    Returns None if the command worked. Exit code 0 means success, 124 means it
    timed out, 127 means the command doesn't exist. Anything else, we search the
    text for clues like "SyntaxError" or "FAILED".
    """
    if timed_out:
        return "timeout"
    if exit_code == 0:
        return None
    if exit_code == 127:
        return "command_not_found"
    for name, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            return name
    return "nonzero_exit"


def summarise_output(text: str) -> str:
    """Pick the one line a person would actually want to read.

    In order: a test result line, then the last error message, then just the
    last line of output.
    """
    pytest = _PYTEST_SUMMARY.findall(text)
    if pytest:
        return _short(pytest[-1])
    ran, result = _UNITTEST_RAN.findall(text), _UNITTEST_RESULT.findall(text)
    if ran and result:
        return _short(f"{ran[-1]} - {result[-1]}")
    exceptions = [m.group(0).strip() for m in _EXCEPTION_LINE.finditer(text)]
    if exceptions:
        return _short(exceptions[-1])
    lines = [line for line in text.splitlines() if line.strip()]
    return _short(lines[-1]) if lines else ""


# Same idea as classify_output, but for when the tool itself failed, (bad file path, unknown tool name, etc.) rather than a command.
def _error_kind(message: str) -> str:
    lowered = message.casefold()
    if "unknown tool" in lowered or "invalid arguments" in lowered or "expecting" in lowered:
        return "bad_call"
    if "relative project path" in lowered:
        return "bad_path"
    for name, pattern in _ERROR_PATTERNS:
        if pattern.search(message):
            return name
    return "tool_error"


def observe(tool_call: dict[str, Any], raw_output: str, duration_ms: int | None = None) -> Observation:
    """Turn one tool call and its result into an Observation.

    Each tool's output looks different, so we handle them one by one: commands
    get checked for errors, file reads say how much was read, and so on.
    """
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    tool = function.get("name") or "unknown"
    arguments = _parse(function.get("arguments", "{}"))
    if not isinstance(arguments, dict):
        arguments = {"raw": arguments}
    common = {"tool": tool, "arguments": arguments, "duration_ms": duration_ms,
              "call_id": tool_call.get("id") if isinstance(tool_call, dict) else None}
    result = _parse(raw_output)

    if isinstance(result, dict) and "error" in result:
        message = str(result["error"]).replace("execute_tool: ", "")
        return Observation(ok=False, error_type=_error_kind(message), summary=_short(message), **common)

    # Commands: the interesting case. Look at both stdout and stderr for errors.
    if tool == "run_command" and isinstance(result, dict):
        combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
        exit_code = result.get("exit_code")
        error_type = classify_output(combined, exit_code, bool(result.get("timed_out")))
        summary = ("Command timed out after 30 seconds" if error_type == "timeout"
                   else summarise_output(combined) or f"exit code {exit_code}")
        return Observation(ok=error_type is None, error_type=error_type, summary=summary,
                           exit_code=exit_code,
                           truncated=bool(result.get("truncated")) or _CUT_MARKER in combined,
                           **common)

    # The other tools either work or raise an error, so if it got here they worked. Just write a short summary.
    if tool == "read_file" and isinstance(result, dict):
        content = result.get("content", "")
        cut = bool(result.get("truncated")) or _CUT_MARKER in content
        return Observation(ok=True, summary=f"Read {len(content)} characters" + (" (cut)" if cut else ""),
                           truncated=cut, **common)
    if tool == "write_file" and isinstance(result, dict):
        return Observation(ok=True, summary=f"Wrote {result.get('written', arguments.get('path'))}", **common)
    if tool == "list_files" and isinstance(result, dict):
        return Observation(ok=True, summary=f"Listed {len(result.get('entries', []))} entries",
                           truncated=bool(result.get("truncated")), **common)
    if tool == "search_memories" and isinstance(result, list):
        return Observation(ok=True, summary=f"Found {len(result)} memories", **common)
    if tool == "forget_memory":
        return Observation(ok=True, summary="Deleted" if result is True else "Memory not found", **common)
    if tool == "save_memory" and isinstance(result, dict):
        return Observation(ok=True, summary=f"Saved #{str(result.get('id', ''))[:8]}", **common)
    return Observation(ok=True, summary="Completed", **common)


def enrich(raw_output: str, observation: Observation) -> str:
    """Add ok / error_type / summary to the top of the tool result.

    The agent still gets everything the tool returned. It just sees our three
    fields first, so it doesn't have to dig through the output to find out
    what went wrong.
    """
    result = _parse(raw_output)
    if not isinstance(result, dict):
        return raw_output
    header = {"ok": observation.ok, "error_type": observation.error_type,
              "summary": observation.summary}
    return json.dumps(header | result)


# Fixing tool output
#
# A few problems with the tools are easier to fix by sitting between them and the sandbox than by editing the tool files.
# ObservedSandbox passes every request through to the real sandbox and tidies up what comes back.



# Shorten long text but keep both ends. The tools only keep the start, which throws away the bit we care about: errors and test results come last.
def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head - 60
    return f"{text[:head]}\n...[{len(text) - head - tail} {_CUT_MARKER}...\n{text[-tail:]}"


class ObservedSandbox:
    """A stand-in for the real sandbox that fixes up what comes back.

    It does four things:
    - Long output keeps its start AND end, so errors at the bottom survive.
    - Odd characters are replaced, so reading a binary file doesn't crash.
    - It remembers the last error message, so "read_file failed" can say why.
    - It skips copying files back to your computer when nothing changed.
      (The orchestrator copies after every file tool, even plain reads,
      which is slow.)

    Anything else (like save_to_host) is passed straight to the real sandbox.
    """

    # The commands read_file (cat) and list_files (python3) use. These only look at files, so running them can't change anything.
    _READ_ONLY = {"cat", "python3"}

    def __init__(self, sandbox: Any, limit: int = OUTPUT_LIMIT) -> None:
        self._sandbox = sandbox
        self._limit = limit
        self.last_error = ""
        self.dirty = True
        self.syncs = 0
        self.skipped_syncs = 0

    # Only copy files back if something could actually have changed.
    def save_to_host(self, project) -> None:
        if not self.dirty:
            self.skipped_syncs += 1
            return
        self._sandbox.save_to_host(project)
        self.syncs += 1
        self.dirty = False

    # Anything we don't handle ourselves goes straight to the real sandbox.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._sandbox, name)

    # Every tool ends up calling this to run something in the sandbox.
    def _execute(self, command, *args, **kwargs):
        # Anything that isn't a plain read might have changed files.
        if not command or command[0] not in self._READ_ONLY:
            self.dirty = True
        result = self._sandbox._execute(command, *args, **kwargs)
        # Don't trim list_files output: it's JSON, and cutting it in half would break it.
        clip = not (command and command[0] == "python3")

        def clean(value):
            if not isinstance(value, (bytes, bytearray)):
                return value
            text = bytes(value).decode("utf-8", errors="replace")
            return (_clip(text, self._limit) if clip else text).encode("utf-8")

        stderr = clean(result.stderr)
        # If it failed, hang on to the error message so we can report it later.
        if result.exit_code != 0:
            text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr or "")
            self.last_error = _short(" ".join(text.split())[-300:], 300) or f"exit code {result.exit_code}"
        return SimpleNamespace(stdout=clean(result.stdout), stderr=stderr, exit_code=result.exit_code)


def _add_error_detail(raw_output: str, sandbox: ObservedSandbox) -> str:
    """Add the real reason to vague tool errors.

    The file tools just say "read_file failed". This adds what the sandbox
    actually said, e.g. "read_file failed: cat: x.py: No such file or directory".
    """
    result = _parse(raw_output)
    if (isinstance(result, dict) and isinstance(result.get("error"), str)
            and result["error"].endswith(" failed") and sandbox.last_error):
        return json.dumps({"error": f"{result['error']}: {sandbox.last_error}"})
    return raw_output


# Noticing when an agent is stuck
#
# Agents sometimes try the same thing over and over and get the same error. this counts how many times in a row that's happened.



# How many times in a row the same thing has failed, and what it was.
@dataclass(frozen=True)
class Streak:
    count: int
    tool: str | None = None
    summary: str = ""


# Two failures count as "the same" if it was the same tool, with the same
# input, giving the same kind of error and the same message.
def _signature(o: dict[str, Any]) -> str:
    return json.dumps([o.get("tool"), o.get("arguments"), o.get("error_type"),
                       (o.get("summary") or "")[:120]], sort_keys=True)


def failure_streak(observations: Iterable[dict[str, Any]]) -> Streak:
    """Count how many times in a row the latest failure has happened.

    We walk backwards from the newest result. As soon as we hit a success or a
    different failure, we stop counting. So any success resets it to 0.
    """
    current = list(observations)
    if not current or current[-1].get("ok"):
        return Streak(0)
    last, count = current[-1], 0
    for o in reversed(current):
        if o.get("ok") or _signature(o) != _signature(last):
            break
        count += 1
    return Streak(count, last.get("tool"), last.get("summary") or "")


# Writing the log
#
# Everything that happens gets written to a log file, one line per event. later, the metrics at the bottom of this file read these logs back.



class Tracer:
    """Writes events to a log file, one file per conversation.

    Each line is one event in JSON, e.g. a tool call or a review result.
    If the log can't be written (disk full, no permission), we print a warning
    once and carry on. Logging should never be the reason a run fails.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        configured = directory if directory is not None else os.getenv("TRACE_DIR")
        self.directory = Path(configured or DEFAULT_TRACE_DIR).expanduser()
        self._warned = False

    # The log file for a conversation. Odd characters are swapped out so the name is always a valid file name.
    def path(self, thread: str) -> Path:
        safe = re.sub(r"[^\w.-]", "_", thread) or "default"
        return self.directory / f"{safe}.jsonl"

    # Add one event to the end of the log, with the time it happened.
    def log(self, thread: str, event: str, **fields: Any) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "thread": thread, "event": event, **fields}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.path(thread).open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, default=str) + "\n")
        except OSError as error:
            if not self._warned:
                print(f"Trace logging disabled: {error}")
                self._warned = True


# Read a log file back in. Any broken lines are skipped.
def read_trace(path: str | Path) -> list[dict[str, Any]]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


# Plugging into the other agents




@dataclass
class Scope:
    """One agent doing one job: a worker on one task, the reviewer, or the coder.

    It keeps that agent's tool results, so we can tell if THIS agent is stuck,
    without mixing in what other agents did.
    """

    thread: str
    turn: int
    agent: str
    task_id: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    stopped: Streak | None = None
    llm_calls: int = 0


# Which agent is running right now. A ContextVar works like a global variable, but each running task gets its own copy, so agents don't overwrite each other.
_SCOPE: ContextVar[Scope | None] = ContextVar("observation_scope", default=None)

# Token counts from the last model call. Filled in when the response comes back, then read when we log that call.
_USAGE: ContextVar[dict[str, int] | None] = ContextVar("observation_usage", default=None)

# Shared state: which logger we're using, which turn each conversation is on, and the baseline coder's scope (it has no task wrapper of its own).
_ACTIVE: dict[str, Any] = {"tracer": None, "turns": {}, "baseline": {}}

# The real functions we replaced, so uninstall() can put them back.
_ORIGINALS: dict[tuple[Any, str], Any] = {}


def _tracer() -> Tracer:
    return _ACTIVE["tracer"] or Tracer()


def _thread() -> str:
    """Which conversation we're in. Used to name the log file.

    LangGraph already turns the project, user and thread ID into a short code,
    so we just use the first 16 characters of that.
    """
    try:
        from langgraph.config import get_config
        return str(get_config().get("configurable", {}).get("thread_id", "default"))[:16]
    except Exception:
        return "default"


def _scope(default_agent: str = "coder") -> Scope:
    """Find out which agent is running right now.

    Workers and the reviewer set this when they start. The baseline coder
    doesn't, so if nothing is set we use one shared "coder" scope per turn.
    """
    scope = _SCOPE.get()
    if scope is not None:
        return scope
    thread = _thread()
    turn = _ACTIVE["turns"].get(thread, 0)
    key = (thread, turn)
    if key not in _ACTIVE["baseline"]:
        _ACTIVE["baseline"][key] = Scope(thread, turn, default_agent)
    return _ACTIVE["baseline"][key]


# Write an event, tagged with which agent and task it came from.
def _log(scope: Scope, event: str, **fields: Any) -> None:
    _tracer().log(scope.thread, event, turn=scope.turn, agent=scope.agent,
                  task_id=scope.task_id, **fields)


# OpenRouter and LangChain name token counts differently. Turn either one into the same shape: prompt_tokens and completion_tokens.
def _usage(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens", raw.get("input_tokens"))
    completion = raw.get("completion_tokens", raw.get("output_tokens"))
    if prompt is None and completion is None:
        return None
    return {"prompt_tokens": int(prompt or 0), "completion_tokens": int(completion or 0)}


# A fake model reply we hand back when we stop an agent. It looks like a normal final answer (no tool calls), so the agent's loop just finishes.
def _stop_reply(streak: Streak) -> dict[str, Any]:
    text = (f"Stopped by observer: {streak.tool} failed the same way {streak.count} times "
            f"in a row ({streak.summary}).")
    return {"text": text, "tool_calls": [], "message": {"role": "assistant", "content": text}}


# Swap a function in another module for our version, and remember the original. Doing it twice does nothing, so we never wrap a wrapper.
def _wrap(module: Any, name: str, make: Callable[[Any], Any]) -> None:
    key = (module, name)
    if key in _ORIGINALS:
        return
    original = getattr(module, name)
    _ORIGINALS[key] = original
    setattr(module, name, make(original))


# Runs every time ANY agent uses a tool. This is the heart of observation.
def _hook_execute_tool(original):
    @functools.wraps(original)
    def execute_tool(tool_call, *args, **kwargs):
        # Clear the old error message so we don't blame this call for the last one.
        sandbox = kwargs.get("sandbox")
        if isinstance(sandbox, ObservedSandbox):
            sandbox.last_error = ""

        # Run the real tool and time it.
        started = time.perf_counter()
        raw = original(tool_call, *args, **kwargs)
        if isinstance(sandbox, ObservedSandbox):
            raw = _add_error_detail(raw, sandbox)
        duration = round((time.perf_counter() - started) * 1000)

        # Work out what happened, remember it for this agent, and log it.
        scope = _scope()
        observation = observe(tool_call, raw, duration)
        scope.observations.append(observation.to_dict())
        _log(scope, "tool", **observation.to_dict())

        # Give the agent the result with our ok / error_type / summary on top. (Memory tools are left alone; their output is used as-is elsewhere.)
        output = enrich(raw, observation) if observation.tool in SPECS else raw

        # Is this agent stuck?
        streak = failure_streak(scope.observations)
        if streak.count == NUDGE_AFTER:
            # 3rd time: add a note to the result telling it to try something else.
            note = (f"This call failed the same way {streak.count} times: {streak.summary}. "
                    "Repeating it will not help. Read the error, inspect the relevant files, "
                    "and try a different approach, or report that the task cannot be done.")
            parsed = _parse(output)
            output = json.dumps(parsed | {NUDGE_NOTE: note}) if isinstance(parsed, dict) else output
            _log(scope, "stuck_nudge", repeats=streak.count, tool=streak.tool, summary=streak.summary)
        elif streak.count >= STOP_AFTER and scope.stopped is None:
            # 5th time: mark it as stopped. The next time it asks the model what to do, send_to_coder below will end its run instead.
            scope.stopped = streak
            _log(scope, "stuck_stop", repeats=streak.count, tool=streak.tool, summary=streak.summary)
        return output
    return execute_tool


# Runs every time a worker, the reviewer or the coder asks the model what to do.
def _hook_send_to_coder(original):
    @functools.wraps(original)
    def send_to_coder(*args, **kwargs):
        scope = _scope()
        if scope.stopped is not None:
            # This agent was stopped for being stuck. Don't ask the model; just reply "stopped" ourselves so the agent's loop finishes normally.
            return _stop_reply(scope.stopped)

        # Ask the model as normal, and log how long it took and how many tokens.
        _USAGE.set(None)
        started = time.perf_counter()
        result = original(*args, **kwargs)
        scope.llm_calls += 1
        message = result.get("message") or {}
        _log(scope, "llm_call",
             latency_ms=round((time.perf_counter() - started) * 1000),
             usage=_usage(result.get("usage")) or _USAGE.get(),
             tool_calls=[c.get("function", {}).get("name") for c in message.get("tool_calls") or []],
             finish_reason=result.get("finish_reason"))
        return result
    return send_to_coder


class _RequestsWithUsage:
    """Lets us see token counts from coder.py's model calls.

    coder.py gets the token count back from OpenRouter but throws it away.
    We swap out coder.py's copy of `requests` for this one. It sends the request
    exactly the same way, but grabs the token count from the reply first.
    Only coder.py is affected, nothing else that uses `requests`.
    """

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def post(self, *args, **kwargs):
        response = self._module.post(*args, **kwargs)
        try:
            _USAGE.set(_usage(response.json().get("usage")))
        except Exception:
            pass
        return response


# Same idea for the planner, which talks to the model through LangChain. We use a version of ChatOpenAI that notes down the token count.
def _hook_chat_openai(original):
    class ChatOpenAIWithUsage(original):  # type: ignore[misc, valid-type]
        def invoke(self, *args, **kwargs):
            response = super().invoke(*args, **kwargs)
            _USAGE.set(_usage(getattr(response, "usage_metadata", None)))
            return response
    ChatOpenAIWithUsage.__name__ = original.__name__
    return ChatOpenAIWithUsage


# Runs when the planner makes a plan. Logs how many tasks it made, how long it took, and if it failed (e.g. the model gave back broken JSON).
def _hook_create_plan(original):
    @functools.wraps(original)
    def create_plan(request, history, *args, **kwargs):
        thread = _thread()
        scope = Scope(thread, _ACTIVE["turns"].get(thread, 0), "planner")
        _USAGE.set(None)
        started = time.perf_counter()
        try:
            plan = original(request, history, *args, **kwargs)
        except Exception as error:
            _log(scope, "plan", ok=False, error=_short(str(error)),
                 latency_ms=round((time.perf_counter() - started) * 1000), usage=_USAGE.get())
            raise
        _log(scope, "plan", ok=True, tasks=len(plan.get("tasks", [])),
             task_ids=[t.get("id") for t in plan.get("tasks", [])],
             latency_ms=round((time.perf_counter() - started) * 1000), usage=_USAGE.get())
        return plan
    return create_plan


# Run something with `scope` set as the current agent, then put things back.
def _run_scoped(scope: Scope, call: Callable[[], Any]):
    token = _SCOPE.set(scope)
    try:
        return call()
    finally:
        _SCOPE.reset(token)


# Runs when a worker does one task from the plan.
#
# Normally if a worker gets stuck or runs out of rounds, the whole run crashes.
# Instead this hands back a "failed" task with the reason. The reviewer then sees it failed, and the orchestrator's repair loop gets a chance to fix it.
def _hook_execute_task(original):
    @functools.wraps(original)
    def execute_task(task, *args, **kwargs):
        thread = _thread()
        scope = Scope(thread, _ACTIVE["turns"].get(thread, 0), "worker", str(task.get("id")))
        started = time.perf_counter()
        failure = None
        try:
            result = _run_scoped(scope, lambda: original(task, *args, **kwargs))
        except RuntimeError as error:
            # Ran out of rounds: treat it as a failed task. Any other error (e.g. the API is down) we let through, so /retry still works.
            if not _ROUND_LIMIT.search(str(error)):
                raise
            failure = f"Stopped by observer: {error}"
            result = None
        # We stopped it for being stuck, so it didn't really finish.
        if scope.stopped is not None:
            failure = _stop_reply(scope.stopped)["text"]
        if failure:
            result = {"task_id": task["id"], "title": task.get("title", ""), "status": "failed",
                      "summary": failure + " The task is not complete."}
        _log(scope, "task_end", status=result.get("status"), title=task.get("title"),
             tool_calls=len(scope.observations), llm_calls=scope.llm_calls,
             duration_ms=round((time.perf_counter() - started) * 1000))
        return result
    return execute_task


# Runs when the reviewer checks the workers' work.
#
# If the reviewer gets stuck, it can't give a proper PASS/FAIL answer and the run would crash. 
#  Instead this counts it as a FAIL ("couldn't check it"), so the repair loop carries on.
def _hook_review_work(original):
    @functools.wraps(original)
    def review_work(*args, **kwargs):
        thread = _thread()
        scope = Scope(thread, _ACTIVE["turns"].get(thread, 0), "reviewer")
        started = time.perf_counter()
        try:
            review = _run_scoped(scope, lambda: original(*args, **kwargs))
        except (RuntimeError, ValueError) as error:
            # Only step in if WE stopped it, or it ran out of rounds. Anything else is a real error, so log it and let it through.
            stuck = scope.stopped is not None
            if not stuck and not _ROUND_LIMIT.search(str(error)):
                _log(scope, "review", status="ERROR", error=_short(str(error)))
                raise
            reason = _stop_reply(scope.stopped)["text"] if stuck else f"Stopped by observer: {error}"
            review = {"status": "FAIL",
                      "feedback": reason + " The reviewer could not verify the work.",
                      "summary": "Review could not be completed."}
        _log(scope, "review", status=review.get("status"), feedback=_short(review.get("feedback", "")),
             tool_calls=len(scope.observations), llm_calls=scope.llm_calls,
             duration_ms=round((time.perf_counter() - started) * 1000))
        return review
    return review_work


def install(tracer: Tracer | None = None) -> None:
    """Plug observation into the other agents' code.

    Swaps each function listed below for our version. You can call this more
    than once; it won't double-wrap anything.
    """
    import agent_roles.coder as coder
    import agent_roles.executor as executor
    import agent_roles.graph as graph
    import agent_roles.openrouter as openrouter
    import agent_roles.planner as planner

    if tracer is not None or _ACTIVE["tracer"] is None:
        _ACTIVE["tracer"] = tracer or Tracer()
    _wrap(graph, "execute_tool", _hook_execute_tool)          # every tool call
    _wrap(graph, "send_to_coder", _hook_send_to_coder)        # model calls (baseline coder)
    _wrap(executor, "send_to_coder", _hook_send_to_coder)     # model calls (workers, reviewer)
    _wrap(graph, "create_plan", _hook_create_plan)            # the planner
    _wrap(graph, "execute_task", _hook_execute_task)          # each worker task
    _wrap(graph, "review_work", _hook_review_work)            # each review
    _wrap(coder, "requests", _RequestsWithUsage)              # token counts (coder)
    _wrap(openrouter, "ChatOpenAI", _hook_chat_openai)        # token counts (planner)
    del planner  # only imported so it's loaded first; we don't use it directly


def uninstall() -> None:
    """Put all the original functions back. Mostly used by the tests."""
    for (module, name), original in _ORIGINALS.items():
        setattr(module, name, original)
    _ORIGINALS.clear()
    _ACTIVE.update(tracer=None, turns={}, baseline={})


# Marking the start and end of each request




# Which request number this is: just count the user's messages so far.
def _count_turn(values: dict[str, Any], update: Any) -> int:
    users = sum(1 for m in values.get("messages", []) if m.get("role") == "user")
    if isinstance(update, dict):
        users += sum(1 for m in update.get("messages", []) if m.get("role") == "user")
    return users


# Look at the finished request and decide how it went.
#   completed  the reviewer passed the work
#   failed     the reviewer still failed it after all the repair attempts
#   answered   the planner just answered a question, no tasks needed
#   stopped    (baseline only) we stopped the coder for being stuck
def _turn_status(values: dict[str, Any]) -> str:
    if "review" in values or "tasks" in values:      # the multi-agent version
        if not values.get("tasks"):
            return "answered"
        if (values.get("review") or {}).get("status") == "PASS":
            return "completed"
        return "failed"
    last = (values.get("messages") or [{}])[-1]      # the single-agent baseline
    if str(last.get("content") or "").startswith("Stopped by observer"):
        return "stopped"
    return "completed"


class ObservedGraph:
    """Wraps the graph so we log when each request starts and ends.

    main.py uses this exactly like the normal graph. Anything we don't handle
    here (like get_state) goes straight through to the real one.
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    # Before the request runs: work out the turn number and log the start.
    def _begin(self, update, config) -> tuple[str, int]:
        thread = str(config.get("configurable", {}).get("thread_id", "default"))[:16]
        try:
            values = self._graph.get_state(config).values or {}
        except Exception:
            values = {}
        turn = _count_turn(values, update)
        _ACTIVE["turns"][thread] = turn
        text = ""
        if isinstance(update, dict) and update.get("messages"):
            text = str(update["messages"][-1].get("content", ""))[:500]
        _tracer().log(thread, "turn_start", turn=turn, text=text, retry=update is None)
        return thread, turn

    # After the request runs: log how it ended (or the error, if it crashed).
    def _end(self, thread: str, turn: int, config, error: Exception | None = None) -> None:
        if error is not None:
            _tracer().log(thread, "turn_end", turn=turn, status="error", error=_short(str(error)))
            return
        try:
            values = self._graph.get_state(config).values or {}
        except Exception:
            values = {}
        _tracer().log(thread, "turn_end", turn=turn, status=_turn_status(values),
                      tasks=len(values.get("tasks") or []),
                      reviews=values.get("review_attempt"),
                      review=(values.get("review") or {}).get("status"))

    # main.py uses stream(); the tests use invoke(). Both get logged the same way.
    def stream(self, update, config, *args, **kwargs):
        thread, turn = self._begin(update, config)
        try:
            yield from self._graph.stream(update, config, *args, **kwargs)
        except Exception as error:
            self._end(thread, turn, config, error)
            raise
        self._end(thread, turn, config)

    def invoke(self, update, config, *args, **kwargs):
        thread, turn = self._begin(update, config)
        try:
            result = self._graph.invoke(update, config, *args, **kwargs)
        except Exception as error:
            self._end(thread, turn, config, error)
            raise
        self._end(thread, turn, config)
        return result


def create_observed_orchestrator(*, checkpointer, store, sandbox, project: Path,
                                 tracer: Tracer | None = None, **kwargs):
    """Use this instead of create_orchestrator_graph to switch observation on.

    It builds the normal multi-agent graph, but with our hooks plugged in and
    the sandbox wrapped. Everything else works exactly the same.
    """
    install(tracer)
    import agent_roles.graph as graph
    sandbox = sandbox if isinstance(sandbox, ObservedSandbox) else ObservedSandbox(sandbox)
    return ObservedGraph(graph.create_orchestrator_graph(
        checkpointer=checkpointer, store=store, sandbox=sandbox, project=project, **kwargs))


def create_observed_coder_graph(*, checkpointer, store, sandbox, project: Path,
                                tracer: Tracer | None = None, **kwargs):
    """Same as above, but for the single-agent baseline (create_coder_graph).

    Use this when running the baseline, so its numbers can be compared
    directly with the multi-agent system's.
    """
    install(tracer)
    import agent_roles.graph as graph
    sandbox = sandbox if isinstance(sandbox, ObservedSandbox) else ObservedSandbox(sandbox)
    return ObservedGraph(graph.create_coder_graph(
        checkpointer=checkpointer, store=store, sandbox=sandbox, project=project, **kwargs))


# Turning the logs into numbers
#
# This is what `python -m observation` runs. It reads the log files, groups the events by request, and works out the numbers for the report.



# Group a log's events by request, and add up the numbers for each one.
def _turn_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: dict[tuple, dict[str, Any]] = {}
    for e in events:
        key = (e.get("thread"), e.get("turn"))
        t = turns.setdefault(key, {"thread": key[0], "turn": key[1], "task": "", "status": "unfinished",
                                   "llm": [], "tools": [], "tasks": [], "reviews": [], "nudges": 0,
                                   "stops": 0})
        kind = e.get("event")
        if kind == "turn_start":
            t["task"] = t["task"] or e.get("text", "")
        elif kind in ("llm_call", "plan"):
            t["llm"].append(e)
        elif kind == "tool":
            t["tools"].append(e)
        elif kind == "task_end":
            t["tasks"].append(e)
        elif kind == "review":
            t["reviews"].append(e)
        elif kind == "stuck_nudge":
            t["nudges"] += 1
        elif kind == "stuck_stop":
            t["stops"] += 1
        elif kind == "turn_end":
            t["status"] = e.get("status", "unknown")

    rows = []
    for t in turns.values():
        tools = t["tools"]
        failures = [i for i, e in enumerate(tools) if not e.get("ok")]
        # A failure counts as "recovered" if the same agent later used the same tool successfully (e.g. tests failed, then passed after a fix).
        recovered = sum(
            any(l.get("ok") and l.get("tool") == tools[i].get("tool")
                and l.get("agent") == tools[i].get("agent") and l.get("task_id") == tools[i].get("task_id")
                for l in tools[i + 1:])
            for i in failures)
        usage = [e.get("usage") or {} for e in t["llm"]]
        rows.append({
            "thread": t["thread"], "turn": t["turn"], "task": (t["task"] or "")[:80],
            "status": t["status"],
            "llm_calls": len(t["llm"]), "tool_calls": len(tools), "tool_failures": len(failures),
            "recovered": recovered,
            "tasks_run": len(t["tasks"]),
            "tasks_failed": sum(1 for e in t["tasks"] if e.get("status") != "complete"),
            "reviews": len(t["reviews"]),
            "review_passes": sum(1 for e in t["reviews"] if e.get("status") == "PASS"),
            "nudges": t["nudges"], "stops": t["stops"],
            "llm_ms": sum(e.get("latency_ms") or 0 for e in t["llm"]),
            "tool_ms": sum(e.get("duration_ms") or 0 for e in tools),
            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
            "errors": [e.get("error_type") or "unknown" for e in tools if not e.get("ok")],
            "_events": t["llm"] + tools,
        })
    return rows


# Work out the overall numbers across every request in the given log files.
def summarise_traces(paths: Iterable[str | Path]) -> dict[str, Any]:
    rows = [row for path in paths for row in _turn_rows(read_trace(path))]

    def rate(part, whole):
        return round(part / whole, 3) if whole else None

    def count(values):
        out: dict[str, int] = {}
        for v in values:
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items(), key=lambda item: -item[1]))

    # Totals for each agent (planner, worker, reviewer), so we can see which, one uses the most time and tokens.
    by_agent: dict[str, dict[str, int]] = {}
    for row in rows:
        for e in row["_events"]:
            agent = by_agent.setdefault(e.get("agent") or "unknown",
                                        {"llm_calls": 0, "tool_calls": 0, "tool_failures": 0, "tokens": 0, "ms": 0})
            if e.get("event") == "tool":
                agent["tool_calls"] += 1
                agent["tool_failures"] += 0 if e.get("ok") else 1
                agent["ms"] += e.get("duration_ms") or 0
            else:
                u = e.get("usage") or {}
                agent["llm_calls"] += 1
                agent["tokens"] += (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
                agent["ms"] += e.get("latency_ms") or 0

    calls = sum(r["tool_calls"] for r in rows)
    failures = sum(r["tool_failures"] for r in rows)
    statuses = count(r["status"] for r in rows)
    times = [r["llm_ms"] + r["tool_ms"] for r in rows]
    prompt = sum(r["prompt_tokens"] for r in rows)
    completion = sum(r["completion_tokens"] for r in rows)
    with_failures = [r for r in rows if r["tool_failures"]]
    reviewed = [r for r in rows if r["reviews"]]
    return {
        "turns": len(rows),
        "status_counts": statuses,
        "success_rate": rate(statuses.get("completed", 0) + statuses.get("answered", 0), len(rows)),
        "tool_calls": calls,
        "tool_success_rate": rate(calls - failures, calls),
        "recovery_rate": rate(sum(r["recovered"] for r in rows), failures),
        "task_recovery_rate": rate(sum(r["status"] == "completed" for r in with_failures), len(with_failures)),
        "first_review_pass_rate": rate(sum(r["reviews"] == 1 and r["review_passes"] == 1 for r in reviewed),
                                       len(reviewed)),
        "mean_reviews": round(statistics.mean(r["reviews"] for r in reviewed), 2) if reviewed else None,
        "worker_tasks": sum(r["tasks_run"] for r in rows),
        "worker_tasks_failed": sum(r["tasks_failed"] for r in rows),
        "observer_nudges": sum(r["nudges"] for r in rows),
        "observer_stops": sum(r["stops"] for r in rows),
        "errors_by_type": count(e for r in rows for e in r["errors"]),
        "mean_llm_calls": round(statistics.mean(r["llm_calls"] for r in rows), 2) if rows else None,
        "median_turn_ms": round(statistics.median(times)) if times else None,
        "tokens": {"prompt": prompt, "completion": completion, "total": prompt + completion},
        "by_agent": by_agent,
        "rows": [{k: v for k, v in r.items() if k != "_events"} for r in rows],
    }


# The command line: `python -m observation`. Prints the numbers, or saves them as CSV (one row per request) or JSON.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise observation traces.")
    parser.add_argument("paths", nargs="*", help="trace files or folders (default: TRACE_DIR)")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--csv", metavar="FILE", help="write one row per user request")
    args = parser.parse_args(argv)

    default = os.getenv("TRACE_DIR") or DEFAULT_TRACE_DIR
    files: list[Path] = []
    for value in args.paths or [str(default)]:
        path = Path(value).expanduser()
        files.extend(sorted(path.glob("*.jsonl")) if path.is_dir() else [path])
    files = [f for f in files if f.is_file()]
    if not files:
        print("No trace files found.", file=sys.stderr)
        return 1

    report = summarise_traces(files)
    if args.csv and report["rows"]:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
            writer.writeheader()
            for row in report["rows"]:
                writer.writerow(row | {"errors": ";".join(row["errors"])})
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    def show(v):
        return "n/a" if v is None else v

    t = report["tokens"]
    print(f"Requests:              {report['turns']}  {report['status_counts']}")
    print(f"Success rate:          {show(report['success_rate'])}")
    print(f"First-review pass:     {show(report['first_review_pass_rate'])}   mean reviews: {show(report['mean_reviews'])}")
    print(f"Worker tasks:          {report['worker_tasks']} ({report['worker_tasks_failed']} failed)")
    print(f"Tool calls:            {report['tool_calls']}   success rate: {show(report['tool_success_rate'])}")
    print(f"Recovery rate:         {show(report['recovery_rate'])}  (failed tool later succeeded)")
    print(f"Task recovery rate:    {show(report['task_recovery_rate'])}  (request with a failure still completed)")
    print(f"Observer:              {report['observer_nudges']} nudges, {report['observer_stops']} stops")
    print(f"Mean LLM calls:        {show(report['mean_llm_calls'])}")
    print(f"Median request time:   {show(report['median_turn_ms'])} ms")
    print(f"Tokens:                {t['total']} ({t['prompt']} prompt, {t['completion']} completion)")
    if report["errors_by_type"]:
        print("Errors by type:        " + ", ".join(f"{k}={v}" for k, v in report["errors_by_type"].items()))
    if report["by_agent"]:
        print("By agent:")
        for name, a in report["by_agent"].items():
            print(f"  {name:<9} llm={a['llm_calls']:<4} tools={a['tool_calls']:<4} "
                  f"failed={a['tool_failures']:<4} tokens={a['tokens']:<7} time={a['ms']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
