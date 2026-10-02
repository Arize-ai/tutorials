"""Send a batch of realistic customer questions through the support agent.

    uv run seed.py              # 100 questions
    uv run seed.py --count 20   # fewer questions
"""

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

from agent import answer, get_provider
from store import ORDERS, TODAY

ORDER_QUESTIONS = [
    "Hi, this is {email}. Where is the order I placed {when}?",
    "Can you check on my order from {when}? My email is {email}.",
    "I ordered something {when} and haven't heard anything. Email: {email}",
    "What's the status of the order I made {when}? I'm {email}.",
]

POLICY_QUESTIONS = [
    "How many days do I have to return a tent?",
    "Can I return a jacket I bought on sale?",
    "Is there a restocking fee if I return my GPS watch?",
    "Do I have to pay for return shipping?",
    "How long does standard shipping take?",
    "Is shipping free?",
    "Do you price match if I find it cheaper somewhere else?",
    "What's the warranty on the Summit 2 Tent?",
    "I've had my headlamp for 3 weeks and don't like it. Can I send it back for a refund?",
    "What's your return policy?",
    "How much is express shipping?",
    "Is the warranty on electronics the same as on tents?",
]

STOCK_QUESTIONS = [
    "Do you have the Summit 2 Tent in stock?",
    "Is the Ridge 30L Daypack available?",
    "How much is the Beacon GPS Watch?",
    "Do you have any Basecamp Stoves left?",
    "Is the Cascade Rain Jacket in stock, and how much is it?",
    "Do you sell water filters?",
]


def describe_date(days_ago: int) -> str:
    """Describe an order date the way a customer would."""
    placed = TODAY - timedelta(days=days_ago)
    options = {0: ["today", "this morning"], 1: ["yesterday"], 2: ["two days ago"], 3: ["three days ago"]}
    options.setdefault(days_ago, [f"on {placed:%B} {placed.day}", "last week" if days_ago <= 7 else "a few weeks ago"])
    # Some customers give an exact date, which the agent handles fine
    options[days_ago].append(placed.isoformat())
    return random.choice(options[days_ago])


def build_questions(count: int) -> list[str]:
    questions = []
    for i in range(count):
        kind = i % 5
        if kind in (0, 1):
            order = random.choice(ORDERS)
            days_ago = (TODAY - order["placed"]).days
            template = random.choice(ORDER_QUESTIONS)
            questions.append(template.format(email=order["email"], when=describe_date(days_ago)))
        elif kind in (2, 3):
            questions.append(random.choice(POLICY_QUESTIONS))
        else:
            questions.append(random.choice(STOCK_QUESTIONS))
    random.shuffle(questions)
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100, help="number of questions to send")
    parser.add_argument("--workers", type=int, default=8, help="questions to run at the same time")
    args = parser.parse_args()

    random.seed(42)
    questions = build_questions(args.count)
    print(f"Sending {len(questions)} questions using {get_provider()}...")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(answer, q): q for q in questions}
        for future in as_completed(futures):
            done += 1
            try:
                future.result()
            except Exception as e:
                failed += 1
                print(f"  request failed: {e}")
            if done % 10 == 0:
                print(f"  {done}/{len(questions)}")

    print(f"Done. {done - failed} answered, {failed} failed.")


if __name__ == "__main__":
    main()
