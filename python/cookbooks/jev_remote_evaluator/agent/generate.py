"""Generate support-response traces and export each prompt to Arize AX."""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from arize.otel import register

EXAMPLES = [
    (
        "I was charged twice for order ORD-104. Please refund the duplicate $49 charge to my original payment method.",
        True,
    ),
    (
        "Please cancel my monthly subscription before the October 1 renewal. My account email is sam@example.com.",
        True,
    ),
    (
        "My unopened package arrived damaged. Please send a replacement to the same address; the order number is ORD-205.",
        True,
    ),
    (
        "I can't log in because I lost access to my old email. Please update my account email to sam@example.com.",
        False,
    ),
    (
        "The app crashes every time I open settings. Please help me fix it.",
        False,
    ),
    (
        "I want to cancel my subscription so it does not renew next month.",
        False,
    ),
]


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env before running this app.")
    for key in ("ARIZE_API_KEY", "ARIZE_SPACE_ID"):
        if not os.getenv(key):
            raise SystemExit(f"Set {key} in .env before running this app.")
    provider = register(
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        project_name=os.getenv("ARIZE_PROJECT_NAME", "jev-remote-evaluator-balanced"),
        batch=False,
    )
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    for index, (prompt, should_resolve) in enumerate(EXAMPLES, 1):
        scenario_instruction = (
            "For this demo scenario, the requested action is available and succeeds. "
            "Complete it and clearly confirm what you did. Do not defer or ask for more information."
            if should_resolve
            else "For this demo scenario, do not take the requested action or provide a solution. "
            "Acknowledge the request and say you will follow up, without resolving it."
        )
        response = client.responses.create(
            model=model,
            instructions=(
                "You are a customer support agent in a fictional demo. Respond naturally and concisely. "
                + scenario_instruction
            ),
            input=prompt,
        )
        print(json.dumps({
            "example": index,
            "intended_label": "yes" if should_resolve else "no",
            "model": model,
            "input": prompt,
            "output": response.output_text,
        }))


if __name__ == "__main__":
    main()
