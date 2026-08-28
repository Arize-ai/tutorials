# Receipt Image Evals

A small Gradio application for testing visual-grounded receipt extraction and preparing image-bearing spans for an Arize AX LLM-as-a-judge evaluator. It discovers numbered PNG images from `images/`.

The extraction model is `gpt-5.4-mini`. Configure `gpt-5.6-luna` as the stronger image-aware judge in AX; the app records that intended judge model on every span but does not make judge calls itself. Set `RECEIPT_JUDGE_MODEL` if your AX AI integration exposes a different judge model.

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

cp .env.example .env
# Edit .env to add your API keys, then load it into the shell:
set -a
source .env
set +a

python receipt_app.py
```

Open the local URL printed by Gradio, select a receipt, and choose **Extract and trace**. The page shows the image, the model result, and the fixture's expected JSON. Use **Inject visual-groundedness error** to replace the merchant and change the total deterministically; this gives the evaluator a reliable `not_grounded` demonstration.

To trace every non-injected fixture without launching the UI:

```bash
python receipt_app.py --batch
```

The app prints one JSON result per fixture. Browse the project named by `ARIZE_PROJECT_NAME` in AX to open its individual traces.

## What is traced

The app registers `arize-otel`, enables OpenInference's OpenAI instrumentor, and creates a `receipt.extract` span. It records:

- the image URL in the input value, `receipt.image.url`, and the OpenInference image-message field
- fixture ID and scenario
- expected structured data and extraction output
- extraction model, intended judge model, and injected-error flag

This makes the data available for a span-level evaluator mapping. Map `receipt_image` to `attributes.input.value` and `extraction` to `attributes.output.value`. The input contains the raw GitHub image URL, so AX and the judge can fetch an image rather than process an embedded base64 payload.

## Evaluator labels

Create a categorical **span-level** LLM-as-a-judge evaluator in AX with these labels:

- `grounded`: the extraction is supported by the visible receipt.
- `not_grounded`: it asserts a merchant, amount, or line item not supported by the image.
- `needs_review`: the image is too degraded or ambiguous to assess confidently.

Use `gpt-5.6-luna` for the judge and start by previewing it on one standard run, one injected-error run, and one degraded fixture. Disable function calling for this evaluator and require its explanation to be valid JSON with `evaluated_extraction` and `judge_result` fields. This keeps the extraction JSON and the judge's label/reason together in AX. Then attach it to a continuous evaluation task. The stronger judge can tell you whether a cheaper extraction model is sufficient only after you calibrate its labels against human annotations from your receipts.

## Cost, privacy, and images

Image-token cost changes with image resolution and the requested detail level (`high` in this example). Measure quality and cost on representative images before increasing resolution or switching models. Do not send customer receipts to a public image host without an approved retention and access policy.

Place images in `images/` as numbered PNG files, beginning with `1.png`. The app discovers them automatically. Public GitHub raw URLs are appropriate for these fictional examples only; use approved private object storage and suitably long-lived signed URLs for real receipts.
