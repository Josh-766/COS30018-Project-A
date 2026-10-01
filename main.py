import os
from pathlib import Path

from memory import MemoryStore, get_relevant_memories

from agent_roles import send_to_coder
from scripts.tools.tool_registry import execute_tool
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

    memory_store = MemoryStore()
    project_id = os.getenv("MEMORY_PROJECT_ID") or str(project)
    context = []

    print("Type 'exit' or 'quit' to stop.")
    print("Memory commands: /remember, /memories, /forget <id>")

    sandbox = create_session(project)
    try:
        while True:
            try:
                text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not text:
                continue

            if text.casefold() in {"exit", "quit"}:
                break

            if text.startswith("/remember "):
                memory_text = text.removeprefix("/remember ").strip()
                try:
                    record = memory_store.add(
                        memory_text,
                        project_id=project_id,
                        source_type="user",
                        reliability=1.0,
                    )
                except ValueError as error:
                    print(f"\nMemory: {error}")
                else:
                    print(f"\nMemory: saved decision #{record.id}")
                continue

            if text == "/memories":
                records = memory_store.list_all(project_id=project_id)
                if not records:
                    print("\nMemory: no saved memories")
                for record in records:
                    print(f"\n[{record.memory_type} #{record.id}] {record.content}")
                continue

            if text.startswith("/forget "):
                memory_id = text.removeprefix("/forget ").strip()
                if not memory_id.isdigit():
                    print("\nMemory: usage: /forget <id>")
                elif memory_store.delete(int(memory_id)):
                    print(f"\nMemory: deleted #{memory_id}")
                else:
                    print(f"\nMemory: #{memory_id} was not found")
                continue

            relevant_memories = get_relevant_memories(
                text,
                store=memory_store,
                project_id=project_id,
                max_tokens=500,
            )

            result = send_to_coder(
                text,
                context=context,
                memories=relevant_memories,
            )

            while result["has_tool_call"]:
                context = result["context"]

                for tool_call in result["tool_calls"]:
                    print(f"\nRunning tool: {tool_call['function']['name']}")
                    output = execute_tool(tool_call, sandbox=sandbox)
                    print(output)
                    context.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": output,
                    })
                    sandbox.save_to_host(project)

                result = send_to_coder(None, context=context)

            context = result["context"]
            print(f"\nCoder: {result['text']}")
    finally:
        sandbox.save_to_host(project)
        sandbox.terminate()


if __name__ == "__main__":
    main()
