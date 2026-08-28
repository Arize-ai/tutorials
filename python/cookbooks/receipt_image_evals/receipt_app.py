"""Gradio receipt-intake app with AX-ready image-grounded trace data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
from openai import OpenAI

from tracing import initialize_tracing
from ui import (
    EXPENSE_INBOX_HEADER,
    INITIAL_EXPENSE_SUMMARY,
    STRUCTURED_RECORD_LABEL,
    SUBMITTED_RECEIPT_LABEL,
    render_expense_summary,
    render_processing_error,
    render_processing_status,
    render_queue_status,
)

ROOT = Path(__file__).parent
IMAGES = ROOT / "images"
DEFAULT_RECEIPT_IMAGE_BASE_URL = (
    "https://raw.githubusercontent.com/Arize-ai/tutorials/"
    "main/python/cookbooks/receipt_image_evals/images"
)
RECEIPT_IMAGE_BASE_URL = os.environ.get(
    "RECEIPT_IMAGE_BASE_URL", DEFAULT_RECEIPT_IMAGE_BASE_URL
).rstrip("/")
IMAGE_PATHS = sorted(IMAGES.glob("*.png"), key=lambda path: int(path.stem))
if not IMAGE_PATHS:
    raise RuntimeError(f"No numbered PNG images found in {IMAGES}")
RECEIPTS_BY_ID = {
    path.stem: {"id": path.stem, "image": path.name}
    for path in IMAGE_PATHS
}
RECEIPT_EXTRACTION_MODEL = os.environ.get("RECEIPT_EXTRACTION_MODEL", "gpt-5.4-mini")
APP_CSS = (ROOT / "app.css").read_text(encoding="utf-8")
EXTRACTION_SYSTEM_PROMPT = """Extract the receipt into the requested JSON schema. Use only what is visually supported by the image.
If a field is unclear, return null or an empty list and set needs_review to true."""

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Set OPENAI_API_KEY before starting the receipt app.")
# Register and instrument before creating the client so every OpenAI call is traced.
TRACER = initialize_tracing(__name__)
OPENAI_CLIENT = OpenAI(api_key=openai_api_key)

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


def image_url(receipt: dict[str, Any]) -> str:
    """Return a fetchable image URL; traces never contain base64 image data."""
    return f"{RECEIPT_IMAGE_BASE_URL}/{receipt['image']}"


def extraction_input(receipt: dict[str, Any]) -> dict[str, str]:
    """Build the sole argument captured as the chain span's input.value."""
    return {
        "system_prompt": EXTRACTION_SYSTEM_PROMPT,
        "image_url": image_url(receipt),
    }


@TRACER.chain(name="receipt.extract")
def model_extract(receipt_input: dict[str, str]) -> dict[str, Any]:
    # The decorator captures receipt_input and the returned JSON as chain I/O.
    response = OPENAI_CLIENT.responses.create(
        model=RECEIPT_EXTRACTION_MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": receipt_input["system_prompt"]},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": receipt_input["image_url"], "detail": "high"},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "receipt_extraction",
                "strict": True,
                "schema": RECEIPT_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)


def run_receipt(receipt_id: str) -> dict[str, Any]:
    receipt = RECEIPTS_BY_ID[receipt_id]
    return model_extract(extraction_input(receipt))


def receipt_choices(processed_ids: list[str]) -> list[tuple[str, str]]:
    processed = set(processed_ids)
    return [
        (f"Receipt #{int(receipt_id):03d} · {'Processed' if receipt_id in processed else 'Pending'}", receipt_id)
        for receipt_id in RECEIPTS_BY_ID
    ]


def queue_status(processed_ids: list[str]) -> str:
    return render_queue_status(total_receipts=len(RECEIPTS_BY_ID), processed_count=len(processed_ids))


def ui_run(receipt_id: str, processed_ids: list[str]):
    processed_ids = processed_ids or []
    try:
        result = run_receipt(receipt_id)
        receipt = RECEIPTS_BY_ID[receipt_id]
        updated_processed = list(dict.fromkeys([*processed_ids, receipt_id]))
        return (
            image_url(receipt),
            render_expense_summary(result),
            json.dumps(result, indent=2),
            render_processing_status(),
            queue_status(updated_processed),
            gr.Dropdown(choices=receipt_choices(updated_processed), value=receipt_id),
            updated_processed,
        )
    except Exception as error:
        receipt = RECEIPTS_BY_ID[receipt_id]
        return (
            image_url(receipt),
            "",
            "",
            render_processing_error(error),
            queue_status(processed_ids),
            gr.Dropdown(choices=receipt_choices(processed_ids), value=receipt_id),
            processed_ids,
        )


def run_batch() -> None:
    for receipt in RECEIPTS_BY_ID.values():
        result = run_receipt(receipt["id"])
        print(json.dumps({"receipt_id": receipt["id"], "result": result}))


def build_app():
    initial_processed: list[str] = []
    choices = receipt_choices(initial_processed)
    with gr.Blocks(title="Expense Inbox") as app:
        gr.HTML(EXPENSE_INBOX_HEADER)
        processed = gr.State(initial_processed)
        queue = gr.HTML(queue_status(initial_processed))
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML(SUBMITTED_RECEIPT_LABEL)
                selected_receipt = gr.Dropdown(choices=choices, value=choices[0][1], show_label=False, elem_id="receipt-selector")
            with gr.Column(scale=1):
                run = gr.Button("Process expense", variant="primary", elem_id="process-expense")
        with gr.Row():
            image = gr.Image(value=image_url(RECEIPTS_BY_ID[choices[0][1]]), show_label=False, buttons=[], type="filepath", height=560, scale=1)
            with gr.Column(scale=1):
                summary = gr.HTML(INITIAL_EXPENSE_SUMMARY)
                gr.HTML(STRUCTURED_RECORD_LABEL)
                output = gr.Textbox(lines=18, interactive=False, show_label=False, elem_id="expense-record")
        status = gr.HTML()
        run.click(ui_run, inputs=[selected_receipt, processed], outputs=[image, summary, output, status, queue, selected_receipt, processed])
        selected_receipt.change(lambda receipt_id: image_url(RECEIPTS_BY_ID[receipt_id]), selected_receipt, image)
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
