# Receipt Image Evals

A demo application to show how to use LLM-as-a-judge evals with images. This example app is for the [Evaluate Receipt Agents with an Image Judge guide](https://arize.com/docs/ax/cookbooks/evaluate/evaluate-receipt-agents-with-image-judge).

This is an example receipt processing application that uses a cheaper model to extract information from the receipt, then a better model for evals to ensure the extraction is working well.

## Prerequisites

- Python 3.10 or later
- An OpenAI API key with access to `gpt-5.4-mini`
- An Arize AX API key and Space ID

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

Open the local URL printed by Gradio, select a submitted receipt from the expense inbox, and choose **Process expense**. The page shows the receipt document, an expense summary, and the structured expense record. Processed receipts move from **Pending** to **Processed** in the in-memory queue; refresh the page to reset it.

To trace every receipt without launching the UI:

```bash
python receipt_app.py --batch
```

The app prints one JSON result per example image. Browse the project named by `ARIZE_PROJECT_NAME` in AX to open its individual traces.

## Evals

Follow the [guide](https://arize.com/docs/ax/cookbooks/evaluate/evaluate-receipt-agents-with-image-judge) to understand the visual-groundedness evaluator. AX runs the stronger judge model through the AI integration you configure in the AX UI; the app traces only the extraction model.

To create the evaluator and its continuous task with the AX CLI, configure an AI integration in AX, then set its ID and run:

```bash
export ARIZE_AI_INTEGRATION_ID="your-ax-ai-integration-id"
./setup_evaluator.sh
```

The script uses `gpt-5.6-luna` by default. Override `RECEIPT_JUDGE_MODEL` when your AX AI integration exposes a different judge model. It is safe to rerun: existing evaluator and task names are reused; an AI integration ID is required only when it needs to create the evaluator.
