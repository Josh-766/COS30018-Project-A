import json
import os
from pathlib import Path
from uuid import uuid4

from memory import (
    MemoryContext,
    forget_memory,
    list_memories,
    open_memory,
    save_memory,
    thread_config,
)

from agent_roles.graph import create_coder_graph, create_orchestrator_graph
from scripts.tools.sandbox_setup import create_session


def set_project_path():
    default = Path.cwd()
    while True:
        try:
            value = input(f"Default directory [{default}] enter another to change: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        project = Path(value.strip('"') or default).expanduser().resolve()
        if project.is_dir():
            return project
        print("That directory does not exist.")


def select_mode():
    print("\nSelect agent system:")
    print("  1. Single agent")
    print("  2. Multi-agent")
    while True:
        try:
            value = input("Selection: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not value:
            return "orchestrator"
        if value == "1":
            return "single"
        if value == "2":
            return "orchestrator"
        print("Enter 1 or 2")


def main():
    project = set_project_path()
    if project is None:
        return

    mode = os.getenv("AGENT_MODE") or select_mode()
    if mode is None:
        return

    context = MemoryContext(
        project_id=os.getenv("MEMORY_PROJECT_ID") or str(project),
        user_id=os.getenv("MEMORY_USER_ID") or "default",
    )
    thread_id = os.getenv("MEMORY_THREAD_ID") or uuid4().hex
    with open_memory() as (checkpointer, store):
        sandbox = create_session(project)
        try:
            if mode == "single":
                print("\nSingle-agent mode")
                graph = create_coder_graph(
                    checkpointer=checkpointer, store=store, sandbox=sandbox, project=project
                )
            else:
                print("\nMulti-agent mode")
                graph = create_orchestrator_graph(
                    checkpointer=checkpointer, store=store, sandbox=sandbox, project=project
                )
            chat(graph, store, context, thread_id)
        finally:
            try:
                sandbox.save_to_host(project)
            finally:
                sandbox.terminate()


def chat(graph, store, context: MemoryContext, thread_id: str):
    # Send only new messages; LangGraph loads previous messages by thread ID.
    print(f"Thread: {thread_id}")
    print("Type 'exit' or 'quit' to stop.")
    print("Memory: /remember <fact>, /memories, /forget <id>")
    print("Conversation: /new, /resume <thread-id>, /retry")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.casefold() in {"exit", "quit"}:
            break

        if text == "/new":
            thread_id = uuid4().hex
            print(f"Thread: {thread_id}")
            continue
        if text.startswith("/resume "):
            thread_id = text.removeprefix("/resume ").strip()
            print(f"Thread: {thread_id}")
            continue

        try:
            if text.startswith("/remember "):
                record = save_memory(store, context, text.removeprefix("/remember "))
                print(f"\nMemory: saved #{record['id']}")
                continue
            if text == "/memories":
                records = list_memories(store, context)
                if not records:
                    print("\nMemory: no saved memories")
                for record in records:
                    print(f"\n[#{record['id']}] {record['content']}")
                continue
            if text.startswith("/forget "):
                memory_id = text.removeprefix("/forget ").strip()
                deleted = forget_memory(store, context, memory_id)
                print(f"\nMemory: #{memory_id} {'deleted' if deleted else 'was not found'}")
                continue
        except ValueError as error:
            print(f"\nMemory: {error}")
            continue

        if text.startswith("/") and text != "/retry":
            print("Unknown command. Use /remember <fact>, /memories, /forget <id>, "
                  "/new, /resume <thread-id>, or /retry.")
            continue

        config = thread_config(context, thread_id)
        pending = bool(graph.get_state(config).next)
        if pending and text != "/retry":
            print("This thread has an unfinished turn. Use /retry to continue it or /new.")
            continue
        if text == "/retry" and not pending:
            print("There is no unfinished turn to retry.")
            continue
        update = None if text == "/retry" else {"messages": [{"role": "user", "content": text}]}
        try:
            for step in graph.stream(update, config, context=context, stream_mode="updates"):
                for node, data in step.items():
                    if node == "planner":
                        print(f"\nPlan: {data.get('plan_summary') or 'Working through the request.'}")
                        for index, task in enumerate(data.get("tasks", []), 1):
                            print(f"  {index}. {task['title']}")
                    elif node == "execute_worker":
                        for result in data.get("task_results", [])[-1:]:
                            print(f"\nWorker [{result['task_id']}]: {result['summary']}")
                    elif node == "reviewer":
                        review = data.get("review", {})
                        print(f"\nReview {review.get('status', 'ERROR')}: {review.get('feedback', '')}")
                    elif node == "finalize":
                        for message in data.get("messages", []):
                            print(f"\nAgent: {message.get('content') or ''}")
                    elif node == "coder":
                        show_step(data)
                    elif node == "tools":
                        print_tool_output(data)
        except Exception as error:
            print(f"\nAgent: {error}\nUse /retry to continue from the last checkpoint or /new.")




def msg_preview(text: str, limit: int = 120):
    #Print the output but cut it off if we go too far
    if not text:
        return ""
    first_line = text.splitlines()[0][:limit]
    if len(first_line) < len(text):
        return first_line + "..."
    return first_line


def show_step(data: dict):

    for message in data.get("messages", []):
        if message.get("role") != "assistant":
            continue

        # Reasoning (if the model returned it)
        for key in ("reasoning"): #need to test the actual response mode better
            reasoning = message.get(key)
            if reasoning:
                print(f"\nThinking")
                for line in reasoning.strip().splitlines():
                    print(f"{line}")

        for toolcall in message.get("tool_calls") or []:
            func = toolcall.get("function", {})
            name = func.get("name", "?")
            args = json.loads(func.get("arguments", "{}"))
            preview = json.dumps(args, ensure_ascii=False)
            print(f"\nTool: {name}")
            for line in msg_preview(preview).splitlines():
                print(f"│ {line}")

        content = message.get("content") or ""
        if content and not message.get("tool_calls"):
            print(f"\n Tool result")
            print(content)


def print_tool_output(data: dict):
    """Print tool call output"""
    for message in data.get("messages", []):
        if message.get("role") == "tool":
            
            result = json.loads(message["content"])
            summary = json.dumps(result)
            summary = str(message.get("content", ""))
            print(f"\nTool Output")
            for line in msg_preview(summary).splitlines():
                print(f"│ {line}")


if __name__ == "__main__":
    main()