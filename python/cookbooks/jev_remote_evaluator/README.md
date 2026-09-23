# Jev Remote Evaluator

This cookbook has two small apps: `agent/` generates customer-support traces with OpenAI and `evaluator/` serves a FastAPI endpoint that evaluates request resolution with TypeSafe Jev. Follow the [Use Jev as a Remote Evaluator guide](https://arize.com/docs/ax/cookbooks/evaluate/jev-remote-evaluator) for the AX walkthrough.

The agent runs six scenarios, three intended to resolve the request and three intended to leave it unresolved. They are intended to produce an even split, but live OpenAI responses and Jev judgments can vary between runs.

## Prerequisites

- Python 3.10 or later
- OpenAI API key with access to the configured model
- Arize AX API key and Space ID
- TypeSafe API key
- A tunneling tool such as `cloudflared` or `ngrok` to expose the local evaluator to AX during the demo
- An Arize AX Enterprise account with remote evaluators enabled

## Configure environment

Each app has its own `.env.example` containing only the values it needs. Copy `agent/.env.example` to `agent/.env` and set its OpenAI and Arize values. Copy `evaluator/.env.example` to `evaluator/.env` and set the TypeSafe key. The agent defaults to `gpt-5.4-mini`; set `OPENAI_MODEL` in `agent/.env` if you want to override it.

## Run the agent

```bash
cd python/cookbooks/jev_remote_evaluator
cp agent/.env.example agent/.env
# Edit agent/.env with your OpenAI and Arize credentials.
python -m venv agent/.venv
source agent/.venv/bin/activate
pip install -r agent/requirements.txt
python agent/generate.py
```

The OpenInference OpenAI instrumentor creates one LLM span per request and captures the request and response as span input/output. Each example is exported promptly. Browse the configured project in AX and confirm its six traces are present.

## Run the Jev evaluator

In a second terminal, configure the evaluator's separate credentials and dependencies:

```bash
cd python/cookbooks/jev_remote_evaluator
cp evaluator/.env.example evaluator/.env
# Edit evaluator/.env with your TypeSafe API key.
python -m venv evaluator/.venv
source evaluator/.venv/bin/activate
pip install -r evaluator/requirements.txt
cd evaluator
set -a; source .env; set +a
uvicorn remote_eval_server:app --host 127.0.0.1 --port 8080
```

Check the health endpoint from another terminal:

```bash
curl http://127.0.0.1:8080/
```

To let AX reach the local server, start a temporary Cloudflare quick tunnel in a third terminal:

```bash
cloudflared tunnel --url http://127.0.0.1:8080
```

Copy its `https://….trycloudflare.com` URL. The unauthenticated quick tunnel is for this temporary demo only; stop it when finished.

## Configure the AX Remote Eval

1. In AX, open **Evaluators → New Evaluator → Create blank Remote Eval**.
2. Set the endpoint to `https://<your-tunnel>.trycloudflare.com/v1/evaluate`. No headers are needed for this demo.
3. Define the input schema with two string fields: `input` and `output`.
4. Map `input` to `attributes.input.value` and `output` to `attributes.output.value`.
5. Select **Test Remote On Span**, choose a generated trace span, and confirm the endpoint returns a label and score.
6. Save the evaluator, create a task for it, select the sample project and the same input/output mappings, then run the task on the project's traces.
7. Open the traces to confirm evaluation results were written.

Jev asks whether the response resolves the request. A probability of `0.5` or higher becomes label `yes`; a lower probability becomes `no`. The probability is returned as `score`.

## Troubleshooting

- `TYPESAFE_API_KEY is not set`: load the key in the evaluator terminal and restart Uvicorn.
- `Jev returned 401`: check that the key is valid and has access.
- `Could not read a valid Jev probability`: Jev returned a missing, malformed, or out-of-range answer; inspect the service output and retry.
- AX cannot reach the endpoint: ensure Uvicorn and the tunnel are running and the URL ends in `/v1/evaluate`.
- Different labels between runs: OpenAI responses and Jev judgments are generated live and can vary.

Stop Uvicorn and `cloudflared` with `Ctrl+C` when finished.
