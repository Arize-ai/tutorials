# Receipt Image Evals

A small Gradio application for testing visual-grounded receipt extraction and preparing image-bearing spans for an Arize AX LLM-as-a-judge evaluator. It discovers numbered PNG images from `images/`.

The extraction model is `gpt-5.4-mini`. Configure `gpt-5.6-terra` as the stronger image-aware judge in AX; the app records that intended judge model on every span but does not make judge calls itself.

## Prerequisites

- Python 3.10 or later
- An OpenAI API key with access to `gpt-5.4-mini`
- An Arize AX API key and Space ID
- HTTPS hosting for the `images/` directory that both OpenAI and Arize AX can fetch

The last requirement is deliberate. `RECEIPT_IMAGE_BASE_URL` is an externally fetchable image reference, rather than a large base64 payload in span attributes. For a local UI-only preview, the PNG files still load from disk. Before publishing an evaluator, confirm in AX that the receipt image field is selectable and rendered from the URL you configured.

## Run it

```bash
cd python/cookbooks/receipt_image_evals
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="..."
export ARIZE_API_KEY="..."
export ARIZE_SPACE_ID="..."
export ARIZE_PROJECT_NAME="receipt-image-evals"
export RECEIPT_IMAGE_BASE_URL="https://your-public-host/receipt-images"

python receipt_app.py
```

Open the local URL printed by Gradio, select a receipt, and choose **Extract and trace**. The page shows the image, the model result, and the fixture's expected JSON. Use **Inject visual-groundedness error** to replace the merchant and change the total deterministically; this gives the evaluator a reliable `not_grounded` demonstration.

To trace every non-injected fixture without launching the UI:

```bash
python receipt_app.py --batch
```

The app prints one JSON result per fixture. Browse the project named by `ARIZE_PROJECT_NAME` in AX to open its individual traces.

For an offline UI or batch smoke test, set `RECEIPT_DEMO_MODE=1`. This returns fixture expected data instead of calling the extraction model, so it is useful for checking the app but cannot measure model quality.

## What is traced

The app registers `arize-otel`, enables OpenInference's OpenAI instrumentor, and creates a `receipt.extract` span. It records:

- the image URL in the input value and `receipt.image.url`
- fixture ID and scenario
- expected structured data and extraction output
- extraction model, intended judge model, and injected-error flag

This makes the data available for a span-level evaluator mapping. For the visual-groundedness prompt, map the receipt image reference to `{receipt_image}`, the output JSON to `{extraction}`, and the scenario to `{scenario}`.

## Evaluator labels

Create a categorical **span-level** LLM-as-a-judge evaluator in AX with these labels:

- `grounded`: the extraction is supported by the visible receipt.
- `not_grounded`: it asserts a merchant, amount, or line item not supported by the image.
- `needs_review`: the image is too degraded or ambiguous to assess confidently.

Use `gpt-5.6-terra` for the judge and start by previewing it on one standard run, one injected-error run, and one degraded fixture. Then attach it to a continuous evaluation task. The stronger judge can tell you whether a cheaper extraction model is sufficient only after you calibrate its labels against human annotations from your receipts.

## Cost, privacy, and images

Image-token cost changes with image resolution and the requested detail level (`high` in this example). Measure quality and cost on representative images before increasing resolution or switching models. Do not send customer receipts to a public image host without an approved retention and access policy.

Place images in `images/` as numbered PNG files, beginning with `1.png`. The app discovers them automatically.
