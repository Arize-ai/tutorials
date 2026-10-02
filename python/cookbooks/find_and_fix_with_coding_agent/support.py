"""Ask the support agent a question.

    uv run support.py "Do you have the Summit 2 tent in stock?"
    uv run support.py            # interactive
"""

import sys

from agent import answer, get_provider


def main() -> None:
    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:])))
        return

    print(f"Trailhead Outfitters support ({get_provider()}). Ctrl+C to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question:
            print(f"Agent: {answer(question)}")


if __name__ == "__main__":
    main()
