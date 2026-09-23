"""Generate support-response traces and export each prompt to Arize AX."""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from arize.otel import register

EXAMPLES = [
    "How can I reset my password if I forgot it?",
    "What does it mean when my tracking says 'out for delivery'?",
    "My package is delayed. What should I check before contacting the carrier?",
    "Please refund the duplicate $49 charge on order ORD-104 to my original payment method.",
    "Please cancel my monthly subscription before it renews next month.",
    "I lost access to my old email. Can you change the email on my account to sam@example.com?",
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
    for index, prompt in enumerate(EXAMPLES, 1):
        response = client.responses.create(
            model=model,
            instructions=(
                "You are a customer support assistant for a fictional online store. "
                "Respond naturally and concisely. You can answer general questions, but you do not "
                "have tools to look up or change customer accounts, orders, or payments. Be clear "
                "about what you can and cannot do."
            ),
            input=prompt,
        )
        print(json.dumps({
            "example": index,
            "model": model,
            "input": prompt,
            "output": response.output_text,
        }))


if __name__ == "__main__":
    main()
