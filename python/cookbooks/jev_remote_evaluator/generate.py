"""Generate support-response traces and export each prompt to Arize AX."""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from arize.otel import register

PROMPTS = [
    "I was charged twice for my order. Please refund the duplicate $49 charge.",
    "My package arrived damaged. Please send a replacement to the same address.",
    "I can't log in because I lost access to my old email. Please update my account email to sam@example.com.",
    "The app keeps crashing when I open settings. Can you fix it or tell me how to stop it?",
    "I want to cancel my subscription and make sure it does not renew next month.",
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
        project_name=os.getenv("ARIZE_PROJECT_NAME", "jev-remote-evaluator"),
        batch=False,
    )
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    tracer = OITracer(provider.get_tracer("jev-remote-evaluator"), config=TraceConfig())
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    for index, prompt in enumerate(PROMPTS, 1):
        response = generate_response(tracer, client, model, prompt)
        print(json.dumps({"example": index, "model": model, "input": prompt, "output": response}))


def generate_response(tracer: OITracer, client: OpenAI, model: str, prompt: str) -> str:
    """The chain span holds the support request and generated response."""
    with tracer.start_as_current_span("support_response") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )
        span.set_attribute("input.value", json.dumps({"input": prompt}))
        response = client.responses.create(
            model=model,
            instructions=(
                "You are a customer support agent. Answer the request directly. "
                "Take an action only when you can actually complete it; otherwise explain what is needed."
            ),
            input=prompt,
        )
        output = response.output_text
        span.set_attribute("output.value", json.dumps({"output": output}))
        return output


if __name__ == "__main__":
    main()
