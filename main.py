from agent_roles import send_to_coder
from memory import get_relevant_memories


def main() -> None:
    memories = []
    context = []

    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue

        relevant_memories = get_relevant_memories(text, memories)


        result = send_to_coder(
            text,
            context=context,
            memories=relevant_memories,
        )

        
        context = result["context"]
        print(f"\nCoder: {result['text']}")


if __name__ == "__main__":
    main()
