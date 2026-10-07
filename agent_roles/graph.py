"""The existing coder/tool loop, with conversation state owned by LangGraph."""

from operator import add
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.config import get_store
from langgraph.func import task
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agent_roles.coder import send_to_coder
from memory import MemoryContext
from scripts.tools.tool_registry import SPECS, execute_tool


class CoderState(TypedDict):
    # Keep provider dictionaries intact, including OpenRouter reasoning_details
    # and raw tool arguments. LangGraph appends and checkpoints each node update.
    messages: Annotated[list[dict[str, Any]], add]


def create_coder_graph(*, checkpointer, store, sandbox, project: Path):
    """Compile once per sandbox; invoke with a scoped thread config and context."""

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
