"""Gradio receipt-intake app with AX-ready image-grounded trace data."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image
from arize.otel import register
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

ROOT = Path(__file__).parent
IMAGES = ROOT / "images"
IMAGE_PATHS = sorted(IMAGES.glob("*.png"), key=lambda path: int(path.stem))
if not IMAGE_PATHS:
    raise RuntimeError(f"No numbered PNG images found in {IMAGES}")
FIXTURE_BY_ID = {
    path.stem: {"id": path.stem, "image": path.name, "scenario": "expense_inbox"}
    for path in IMAGE_PATHS
}
EXTRACTION_MODEL = "gpt-5.4-mini"
# Used when configuring the AX evaluator, not by this app. Set this to the
# model exposed by the selected AX AI integration.
JUDGE_MODEL = os.environ.get("RECEIPT_JUDGE_MODEL", "gpt-5.6-luna")

APP_CSS = """
.gradio-container { max-width: 1120px !important; background: #f7f9fc; padding-top: 24px !important; }
.app-header { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 10px; padding: 14px 0 30px; }
.brand { font-size: 28px; font-weight: 700; color: #24405f; letter-spacing: -0.03em; }
.brand span { color: #4f7cac; }
.eyebrow { color: #718096; font-size: 14px; margin-top: 4px; }
.connection { background: #eaf2f8; color: #426b8f; border-radius: 999px; padding: 7px 12px; font-size: 13px; font-weight: 600; }
.workspace { background: #ffffff; border: 1px solid #e4ebf2; border-radius: 18px; padding: 28px; box-shadow: 0 8px 30px rgba(68, 96, 124, .08); text-align: center; }
.section-title { font-size: 20px; font-weight: 700; color: #35536f; margin-bottom: 6px; }
.section-copy { color: #718096; font-size: 14px; margin-bottom: 18px; }
.queue-status { color: #5f7b94; font-size: 13px; font-weight: 600; margin: -8px 0 16px; }
.expense-summary { border: 1px solid #e4ebf2; border-radius: 14px; padding: 22px; background: #fbfdff; text-align: center; }
.expense-summary h3 { margin: 0 0 8px; color: #35536f; font-size: 20px; }
.expense-total { font-size: 30px; font-weight: 700; color: #426b8f; margin: 4px 0 18px; }
.expense-meta { display: flex; justify-content: center; gap: 30px; color: #718096; font-size: 13px; }
.expense-meta strong { display: block; color: #4a6881; font-size: 14px; margin-top: 3px; }
.review-badge { display: inline-block; margin-top: 18px; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.review-ok { background: #e7f4ed; color: #4b7d60; }
.review-needed { background: #fff3df; color: #a36f32; }
.trace-status { color: #6a849a; font-size: 13px; padding-top: 16px; text-align: center; }
.gr-button-primary { background: #5f87ae !important; border-color: #5f87ae !important; }
"""

RECEIPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "merchant": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"name": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["name", "amount"],
            },
        },
        "subtotal": {"type": ["number", "null"]},
        "tax": {"type": ["number", "null"]},
        "tip": {"type": ["number", "null"]},
        "total": {"type": ["number", "null"]},
        "needs_review": {"type": "boolean"},
    },
    "required": ["merchant", "currency", "items", "subtotal", "tax", "tip", "total", "needs_review"],
}


def image_url(fixture: dict[str, Any]) -> str:
    """Return a public URL, or a compact inline image for local runs and traces."""
    base_url = os.environ.get("RECEIPT_IMAGE_BASE_URL")
    if base_url:
        return f"{base_url.rstrip('/')}/{fixture['image']}"

    # Keep trace payloads manageable while still giving the OpenAI request and
    # Arize span a rendered image that can be inspected without a public host.
    with Image.open(IMAGES / fixture["image"]) as source:
        image = source.convert("RGB")
        image.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def init_tracing():
    required = ("ARIZE_API_KEY", "ARIZE_SPACE_ID")
    if not all(os.environ.get(name) for name in required):
        return None
    provider = register(
        project_name=os.environ.get("ARIZE_PROJECT_NAME", "receipt-image-evals"),
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        batch=True,
    )
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    return provider.get_tracer(__name__)


TRACER = init_tracing()


def extraction_prompt(fixture: dict[str, Any]) -> str:
    return f"""Extract the receipt into the requested JSON schema. Use only what is visually supported by the image.
If a field is unclear, return null or an empty list and set needs_review to true.
This is fixture {fixture['id']} with scenario {fixture['scenario']}."""


def model_extract(fixture: dict[str, Any]) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before extracting a receipt.")
    client = OpenAI()
    response = client.responses.create(
        model=os.environ.get("RECEIPT_EXTRACTION_MODEL", EXTRACTION_MODEL),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": extraction_prompt(fixture)},
                    {"type": "input_image", "image_url": image_url(fixture), "detail": "high"},
                ],
            }
        ],
        text={"format": {"type": "json_schema", "name": "receipt_extraction", "strict": True, "schema": RECEIPT_SCHEMA}},
    )
    return json.loads(response.output_text)


def run_fixture(fixture_id: str) -> tuple[dict[str, Any], str]:
    fixture = FIXTURE_BY_ID[fixture_id]
    attrs = {
        "openinference.span.kind": "CHAIN",
        "input.value": json.dumps({"prompt": extraction_prompt(fixture), "image_url": image_url(fixture)}),
        "input.mime_type": "application/json",
        "receipt.fixture_id": fixture_id,
        "receipt.scenario": fixture["scenario"],
        "receipt.image.url": image_url(fixture),
        "receipt.extraction_model": os.environ.get("RECEIPT_EXTRACTION_MODEL", EXTRACTION_MODEL),
        "receipt.judge_model": JUDGE_MODEL,
    }
    if TRACER is None:
        result = model_extract(fixture)
        return result, "Tracing is disabled: set ARIZE_API_KEY and ARIZE_SPACE_ID."
    with TRACER.start_as_current_span("receipt.extract", attributes=attrs) as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
        try:
            result = model_extract(fixture)
            span.set_attribute("output.value", json.dumps(result))
            span.set_attribute("output.mime_type", "application/json")
            span.set_status(Status(StatusCode.OK))
            return result, f"Extraction complete. Trace ID: {trace_id}"
        except Exception as error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            raise


def expense_summary(result: dict[str, Any]) -> str:
    currency = escape(str(result.get("currency") or "—"))
    merchant = escape(str(result.get("merchant") or "Merchant pending review"))
    total = result.get("total")
    formatted_total = f"{currency} {total:,.2f}" if isinstance(total, (int, float)) else "Amount pending review"
    item_count = len(result.get("items") or [])
    review_needed = result.get("needs_review", False)
    review_class = "review-needed" if review_needed else "review-ok"
    review_text = "Review needed" if review_needed else "Ready for review"
    return f"""
    <div class="expense-summary">
      <h3>{merchant}</h3>
      <div class="expense-total">{formatted_total}</div>
      <div class="expense-meta">
        <div><span>Expense type</span><strong>Receipt</strong></div>
        <div><span>Line items</span><strong>{item_count}</strong></div>
        <div><span>Currency</span><strong>{currency}</strong></div>
      </div>
      <span class="review-badge {review_class}">{review_text}</span>
    </div>
    """


def receipt_choices(processed_ids: list[str]) -> list[tuple[str, str]]:
    processed = set(processed_ids)
    return [
        (f"Receipt #{int(fixture_id):03d} · {'Processed' if fixture_id in processed else 'Pending'}", fixture_id)
        for fixture_id in FIXTURE_BY_ID
    ]


def queue_status(processed_ids: list[str]) -> str:
    processed_count = len(processed_ids)
    pending_count = len(FIXTURE_BY_ID) - processed_count
    return f"<div class=\"queue-status\">{pending_count} pending · {processed_count} processed</div>"


def ui_run(fixture_id: str, processed_ids: list[str]):
    processed_ids = processed_ids or []
    try:
        result, status = run_fixture(fixture_id)
        fixture = FIXTURE_BY_ID[fixture_id]
        updated_processed = list(dict.fromkeys([*processed_ids, fixture_id]))
        return (
            str(IMAGES / fixture["image"]),
            expense_summary(result),
            json.dumps(result, indent=2),
            f"<div class=\"trace-status\">{escape(status)} · Receipt moved to Processed.</div>",
            queue_status(updated_processed),
            gr.Dropdown(choices=receipt_choices(updated_processed), value=fixture_id),
            updated_processed,
        )
    except Exception as error:
        fixture = FIXTURE_BY_ID[fixture_id]
        return (
            str(IMAGES / fixture["image"]),
            "",
            "",
            f"<div class=\"trace-status\">Processing failed: {escape(str(error))}</div>",
            queue_status(processed_ids),
            gr.Dropdown(choices=receipt_choices(processed_ids), value=fixture_id),
            processed_ids,
        )


def run_batch() -> None:
    for fixture in FIXTURE_BY_ID.values():
        result, _ = run_fixture(fixture["id"])
        print(json.dumps({"fixture_id": fixture["id"], "result": result}))
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


def build_app():
    initial_processed: list[str] = []
    choices = receipt_choices(initial_processed)
    with gr.Blocks(title="Expense Inbox") as app:
        gr.HTML("""
        <div class="app-header">
          <div><div class="brand">Northstar <span>Expenses</span></div><div class="eyebrow">Receipt inbox · AI-assisted expense processing</div></div>
          <div class="connection">● Tracing to Arize AX</div>
        </div>
        """)
        with gr.Group(elem_classes="workspace"):
            gr.HTML("<div class=\"section-title\">Expense inbox</div><div class=\"section-copy\">Select a submitted receipt and create a structured expense record.</div>")
            processed = gr.State(initial_processed)
            queue = gr.HTML(queue_status(initial_processed))
            with gr.Row():
                fixture = gr.Dropdown(choices=choices, value=choices[0][1], label="Submitted receipt", scale=3)
                run = gr.Button("Process expense", variant="primary", scale=1)
            with gr.Row():
                image = gr.Image(value=str(IMAGES / FIXTURE_BY_ID[choices[0][1]]["image"]), label="Receipt document", type="filepath", height=560, scale=1)
                with gr.Column(scale=1):
                    summary = gr.HTML("<div class=\"expense-summary\"><h3>Expense details</h3><div class=\"section-copy\">Process a receipt to create an expense record.</div></div>")
                    output = gr.Code(label="Structured expense record", language="json", lines=18)
            status = gr.HTML()
        run.click(ui_run, inputs=[fixture, processed], outputs=[image, summary, output, status, queue, fixture, processed])
        fixture.change(lambda fixture_id: str(IMAGES / FIXTURE_BY_ID[fixture_id]["image"]), fixture, image)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Trace every numbered image, then exit.")
    args = parser.parse_args()
    if args.batch:
        run_batch()
    else:
        build_app().launch(
            server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
            server_port=int(os.environ.get("PORT", "7860")),
            css=APP_CSS,
        )
