from operator import add
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.config import get_store
from langgraph.func import task
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agent_roles.coder import send_to_coder
from agent_roles.executor import AgentExecutionError, execute_task
from agent_roles.planner import create_plan
from agent_roles.reviewer import review_work
from memory import MemoryContext
from scripts.tools.tool_registry import SPECS, execute_tool


class CoderState(TypedDict):
    # Keep provider dictionaries intact, including OpenRouter reasoning_details
    # and raw tool arguments. LangGraph appends and checkpoints each node update.
    messages: Annotated[list[dict[str, Any]], add]


class WorkflowState(TypedDict, total=False):
    # Conversation plus the orchestrator's checkpointed blackboard.

    messages: Annotated[list[dict[str, Any]], add]
    request: str
    plan_summary: str
    tasks: list[dict[str, Any]]
    active_tasks: list[dict[str, Any]]
    task_results: list[dict[str, Any]]
    work_history: list[dict[str, Any]]
    task_index: int
    review_attempt: int
    review_feedback: str
    review: dict[str, str]


def create_coder_graph(*, checkpointer, store, sandbox, project: Path):
    # Compile once per sandbox; invoke with a scoped thread config and context.

    @task
    def run_tool(tool_call, memory_context):
        # Retain completed results if a later call or sandbox sync fails, so a
        # checkpoint retry does not execute a completed command a second time.
        return execute_tool(
            tool_call,
            sandbox=sandbox,
            store=get_store(),
            memory_context=memory_context,
        )

    def coder(state: CoderState):
        result = send_to_coder(None, context=state["messages"])
        return {"messages": [result["message"]]}

    def tools(state: CoderState, runtime: Runtime[MemoryContext]):
        messages = []
        for tool_call in state["messages"][-1].get("tool_calls", []):
            output = run_tool(tool_call, runtime.context).result()
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": output,
            })
            if tool_call.get("function", {}).get("name") in SPECS:
                sandbox.save_to_host(project)
        return {"messages": messages}

    def next_node(state: CoderState):
        return "tools" if state["messages"][-1].get("tool_calls") else END

    builder = StateGraph(CoderState, context_schema=MemoryContext)
    builder.add_node("coder", coder)
    builder.add_node("tools", tools)
    builder.add_edge(START, "coder")
    builder.add_conditional_edges("coder", next_node, ["tools", END])
    builder.add_edge("tools", "coder")
    return builder.compile(checkpointer=checkpointer, store=store)


def create_orchestrator_graph(
    *, checkpointer, store, sandbox, project: Path, max_rework_attempts: int = 2,
    on_event=None,
):
    # Plan, execute workers in order, review, and route bounded repair passes.

    @task
    def run_tool(tool_call, memory_context):
        # LangGraph caches completed tool tasks across a checkpoint retry.
        return execute_tool(
            tool_call,
            sandbox=sandbox,
            store=get_store(),
            memory_context=memory_context,
        )

    def tool_runner(runtime: Runtime[MemoryContext]):
        def call(tool_call):
            output = run_tool(tool_call, runtime.context).result()
            if tool_call.get("function", {}).get("name") in SPECS:
                sandbox.save_to_host(project)
            return output
        return call

    def planner(state: WorkflowState):
        request = next(
            (message.get("content", "") for message in reversed(state["messages"])
             if message.get("role") == "user"),
            "",
        )
        plan = create_plan(request, state["messages"])
        return {
            "request": request,
            "plan_summary": plan["summary"],
            "tasks": plan["tasks"],
            "active_tasks": plan["tasks"],
            "task_results": [],
            "work_history": [],
            "task_index": 0,
            "review_attempt": 0,
            "review_feedback": "",
            "review": {},
        }

    def execute_worker(state: WorkflowState, runtime: Runtime[MemoryContext]):
        index = state["task_index"]
        task_contract = state["active_tasks"][index]
        statuses = {item["task_id"]: item["status"] for item in state["task_results"]}
        blocked = [
            dependency for dependency in task_contract.get("depends_on", [])
            if statuses.get(dependency) != "complete"
        ]
        if blocked:
            result = {
                "task_id": task_contract["id"],
                "title": task_contract["title"],
                "status": "skipped",
                "summary": "Dependencies not complete: " + ", ".join(blocked),
            }
        else:
            result = execute_task(
                task_contract,
                request=state["request"],
                history=state["messages"],
                previous_results=state["work_history"],
                run_tool=tool_runner(runtime),
                review_feedback=state.get("review_feedback", ""),
                on_event=on_event,
            )
        return {
            "task_results": state["task_results"] + [result],
            "work_history": state["work_history"] + [result],
            "task_index": index + 1,
        }

    def reviewer(state: WorkflowState, runtime: Runtime[MemoryContext]):
        try:
            assessment = review_work(
                state["request"],
                state["tasks"],
                state["work_history"],
                history=state["messages"],
                run_tool=tool_runner(runtime),
                on_event=on_event,
            )
        except (AgentExecutionError, ValueError) as error:
            assessment = {"status": "FAIL", "feedback": f"Review failed: {error}", "summary": ""}
        # Current-pass failures must not become a successful final result, even
        # if the reviewer overlooks them. A later repair pass can resolve them.
        unfinished = [item for item in state["task_results"] if item["status"] != "complete"]
        if unfinished:
            assessment = {
                **assessment,
                "status": "FAIL",
                "feedback": assessment["feedback"] + "\nUnfinished tasks:\n" + "\n".join(
                    f"{item['task_id']} ({item['status']}): {item['summary']}"
                    for item in unfinished
                ),
            }
        return {
            "review": assessment,
            "review_attempt": state.get("review_attempt", 0) + 1,
            "review_feedback": assessment["feedback"],
        }

    def rework(state: WorkflowState):
        attempt = state["review_attempt"]
        task = {
            "id": f"review-fix-{attempt}",
            "title": "Address reviewer findings",
            "description": (
                "Inspect and fix the reviewer findings below. Preserve completed work, "
                "finish any failed, incomplete or skipped tasks from the original request, "
                "then run the requested focused checks.\n\n" + state["review_feedback"]
            ),
            "depends_on": [],
            "files": [],
        }
        return {"active_tasks": [task], "task_results": [], "task_index": 0}

    def finalize(state: WorkflowState):
        review = state.get("review", {})
        if not state.get("tasks"):
            answer = state.get("plan_summary") or "No project work was needed."
        elif review.get("status") == "PASS":
            answer = review.get("summary") or review.get("feedback") or "Work completed and reviewed."
        else:
            answer = (
                "The reviewer still found issues after the allowed repair attempts. "
                + str(review.get("feedback", "Review did not pass."))
            )
        return {"messages": [{"role": "assistant", "content": answer}]}

    def after_plan(state: WorkflowState):
        return "execute_worker" if state.get("active_tasks") else "finalize"

    def after_worker(state: WorkflowState):
        return (
            "execute_worker"
            if state["task_index"] < len(state["active_tasks"])
            else "reviewer"
        )

    def after_review(state: WorkflowState):
        if state["review"].get("status") == "PASS":
            return "finalize"
        if state["review_attempt"] <= max_rework_attempts:
            return "rework"
        return "finalize"

    builder = StateGraph(WorkflowState, context_schema=MemoryContext)
    builder.add_node("planner", planner)
    builder.add_node("execute_worker", execute_worker)
    builder.add_node("reviewer", reviewer)
    builder.add_node("rework", rework)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", after_plan)
    builder.add_conditional_edges("execute_worker", after_worker)
    builder.add_conditional_edges("reviewer", after_review)
    builder.add_edge("rework", "execute_worker")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, store=store)
