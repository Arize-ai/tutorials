# Jev Remote Evaluator

This cookbook creates sample customer support traces in Arize AX, then evaluates whether each response resolves the request using TypeSafe Jev as an Arize remote evaluator. Follow the [Use Jev as a Remote Evaluator guide](https://arize.com/docs/ax/cookbooks/evaluate/jev-remote-evaluator) for the full walkthrough.

The generator calls a live OpenAI model, `gpt-5.4-mini` by default. Model responses and Jev judgments can vary between runs, so use the results to demonstrate the workflow rather than as fixed expected labels.

## Prerequisites

- Python 3.10 or later
- OpenAI API key with access to the configured model
- Arize AX API key and Space ID
- TypeSafe API key
- A tunneling tool such as `cloudflared` or `ngrok` to expose the local service to AX during the demo
- An Arize AX Enterprise account with remote evaluators enabled

## Generate traces

```bash
cd python/cookbooks/jev_remote_evaluator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with OpenAI, Arize, and TypeSafe credentials.
python generate.py
```

The batch app runs several baked-in support requests through OpenAI and prints each request/response. The OpenInference OpenAI instrumentor creates one LLM span per prompt and captures the request and response as span input/output. Tracing exports promptly so each example appears in the configured AX project (`ARIZE_PROJECT_NAME`, default `jev-remote-evaluator`). Browse that project in AX and confirm the traces are present before continuing.

Set `OPENAI_MODEL` in `.env` to use another available model. Configure `TYPESAFE_API_KEY` for the local evaluator service.

## Start and test the evaluator

In a second terminal, activate the environment and load the same `.env` file (or export `TYPESAFE_API_KEY` there):

```bash
cd python/cookbooks/jev_remote_evaluator
source .venv/bin/activate
set -a; source .env; set +a
uvicorn remote_eval_server:app --host 127.0.0.1 --port 8080
```

Check the health endpoint:

```bash
curl http://127.0.0.1:8080/
```

It reports whether the evaluator process has a TypeSafe key. AX cannot reach localhost, so in a third terminal start a temporary Cloudflare quick tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:8080
```

Copy the generated `https://….trycloudflare.com` URL. This unauthenticated quick tunnel is for a temporary demo only; stop it when finished.

## Configure the AX Remote Eval

1. In AX, open **Evaluators → New Evaluator → Create blank Remote Eval**.
2. Set the endpoint to `https://<your-tunnel>.trycloudflare.com/v1/evaluate`. No headers are needed for this demo.
3. Define the input schema with two string fields: `input` and `output`.
4. Map `input` to `attributes.input.value` and `output` to `attributes.output.value`.
5. Select **Test Remote On Span**, choose a generated trace span, and confirm the endpoint returns a label and score.
6. Save the evaluator, create a task for it, select the sample project and the same input/output mappings, then run the task on the project's traces.
7. Open the traces to confirm evaluation results were written.

Jev asks whether the response resolves the request. It returns a Noul probability: `0.5` or above becomes label `yes`; below `0.5` becomes `no`. The probability is returned as `score`.

## Troubleshooting

- `TYPESAFE_API_KEY is not set`: load the key in the evaluator terminal and restart Uvicorn.
- `Jev returned 401`: check that the key is valid and has access.
- `Could not read a valid Jev probability`: Jev returned a missing, malformed, or out-of-range answer; inspect the service output and retry.
- AX cannot reach the endpoint: ensure Uvicorn and the tunnel are running and the URL ends in `/v1/evaluate`.
- Different labels between runs: the OpenAI response and Jev judgment are generated live and can vary.

Stop Uvicorn and `cloudflared` with `Ctrl+C` when finished.
