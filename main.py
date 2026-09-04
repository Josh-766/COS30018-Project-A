from agent_roles import send_to_coder
from memory import get_relevant_memories
from scripts.tools.tool_registry import execute_tool


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

        
        while result["has_tool_call"]:
            context = result["context"]

            for tool_call in result["tool_calls"]:
                print(f"\nRunning tool: {tool_call['function']['name']}")
                output = execute_tool(tool_call)
                print(output)
                context.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": output,
                })

            result = send_to_coder(None, context=context)

        context = result["context"]
        print(f"\nCoder: {result['text']}")


if __name__ == "__main__":
    main()
