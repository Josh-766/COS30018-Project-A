from memory import MemoryStore, get_relevant_memories


def main() -> None:
    memory_store = MemoryStore()
    context = []

    print("Type 'exit' or 'quit' to stop.")
    print("Memory commands: /remember, /memories, /forget <id>")

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
                record = memory_store.add(memory_text)
            except ValueError as error:
                print(f"\nMemory: {error}")
            else:
                print(f"\nMemory: saved decision #{record.id}")
            continue

        if text == "/memories":
            records = memory_store.list_all()
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

        relevant_memories = get_relevant_memories(text, store=memory_store)

        # Keep memory-only commands usable before optional API dependencies are loaded.
        from agent_roles import send_to_coder

        result = send_to_coder(
            text,
            context=context,
            memories=relevant_memories,
        )

        context = result["context"]
        print(f"\nCoder: {result['text']}")


if __name__ == "__main__":
    main()
