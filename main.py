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

from agent_roles.graph import create_coder_graph
from scripts.tools.sandbox_setup import create_session


def set_project_path() -> Path | None:
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


def main() -> None:
    project = set_project_path()
    if project is None:
        return

    context = MemoryContext(
        project_id=os.getenv("MEMORY_PROJECT_ID") or str(project),
        user_id=os.getenv("MEMORY_USER_ID") or "default",
    )
    thread_id = os.getenv("MEMORY_THREAD_ID") or uuid4().hex
    with open_memory() as (checkpointer, store):
        sandbox = create_session(project)
        try:
            graph = create_coder_graph(
                checkpointer=checkpointer, store=store, sandbox=sandbox, project=project
            )
            chat(graph, store, context, thread_id)
        finally:
            try:
                sandbox.save_to_host(project)
            finally:
                sandbox.terminate()


def chat(graph, store, context: MemoryContext, thread_id: str) -> None:
    """Send only new messages; LangGraph loads previous messages by thread ID."""
    print("Type 'exit' or 'quit' to stop.")
    print("Memory: /remember <fact>, /memories, /forget <id>")
    print("Conversation: /new, /resume <thread-id>, /retry")
    print(f"Thread: {thread_id}")

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
                for message in step.get("coder", {}).get("messages", []):
                    for call in message.get("tool_calls", []):
                        print(f"\nRunning tool: {call['function']['name']}")
                    if not message.get("tool_calls"):
                        print(f"\nCoder: {message.get('content') or ''}")
                for message in step.get("tools", {}).get("messages", []):
                    print(message["content"])
        except Exception as error:
            print(f"\nCoder: {error}\nUse /retry to continue from the last checkpoint or /new.")


if __name__ == "__main__":
    main()
