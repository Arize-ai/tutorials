# Testing

`scripts/verify.py` runs the example apps against real Arize AX and Dynatrace backends and checks what each backend stored for one request. It is the test for everything in this kit: the collector config, the Python in-code setup, and the Node in-code setup.

## What you need

- Everything in the README quickstart: `.env` filled in, the Python virtual environment, and `npm install` in `example-ts/` for the Node scenario.
- The [`ax` CLI](https://arize.com/docs/ax), logged in with an API key that can read the space in `ARIZE_SPACE_ID`. If you keep several `ax` profiles, set `AX_PROFILE` to the one to use; the script switches to it and switches back when it finishes.
- `DYNATRACE_QUERY_TOKEN`: a Dynatrace platform token with `storage:logs:read`, `storage:metrics:read`, `storage:spans:read`, and `storage:buckets:read`. The script reads spans, logs, and metrics back through the Grail query API.
- For the collector scenarios, a collector binary in `bin/`. `scripts/run-collector.sh` downloads one on first use (`DISTRO=contrib` for `otelcol-contrib`). The script starts and stops the collector itself, so nothing may be listening on port 13133.

Settings are read from `.env` at the kit root. Values in `.env` take precedence over variables exported in your shell.

## Run it

```bash
.venv/bin/python scripts/verify.py                     # every scenario
.venv/bin/python scripts/verify.py collector incode    # some scenarios
COLLECTOR_BIN=otelcol-contrib .venv/bin/python scripts/verify.py collector collector-nochain collector-nested
```

Each scenario prints one line per check, a pass count, the span names each backend stored, and a DQL query for the trace. The full results, including every span name, parent, and log line, are written to `out/results-<scenario>.json`. The exit code is 1 if any check fails.

A collector scenario takes about a minute: the script waits `ARIZE_ROOT_DECISION_WAIT` (30 s by default) after the request so the root span branch has decided, then polls both backends until the data is there.

## Scenarios

| Scenario | App | Path | Arize root expected |
| --- | --- | --- | --- |
| `collector` | Python, request span tagged as a CHAIN (`AI_ROOT=server`) | collector | `POST /ask` |
| `collector-nochain` | Python, request span not tagged (`AI_ROOT=none`) | collector | `POST /ask`, tagged by the collector |
| `collector-nested` | Python, not tagged, `triage_agent` wrapping `support_agent` (`NESTED_AGENTS=1`) | collector | `POST /ask`, with `triage_agent` under it and `support_agent` under `triage_agent` |
| `incode` | Python, not tagged | `OpenInferenceSpanProcessor` | `support_agent` |
| `incode-nested` | Python, not tagged, nested agents | `OpenInferenceSpanProcessor` | `triage_agent`, with `support_agent` under it |
| `incode-ts` | Node, Vercel AI SDK v7 | `ArizeRootProcessor` | `invoke_agent ...` |

The collector scenarios use the Dynatrace distribution if `bin/dynatrace-otel-collector` exists, otherwise `otelcol-contrib`. `COLLECTOR_BIN` picks one.

## Checks

Every scenario:

| Check | What it proves |
| --- | --- |
| app exported without errors | No exporter reported a failed request (wrong endpoint, auth, or payload) |
| arize: all AI spans present | Every AI span of the request reached Arize: agents, LLM calls, and both tools, including the one that uses only `gen_ai.*` attributes |
| arize: no database or HTTP client spans | The Arize filter dropped the APM spans |
| arize: one root, no dangling parents | The trace has exactly one root in Arize, it is the expected span, and every other span's parent is also in Arize |
| arize: trace listed in Traces | `ax traces list` returns the trace, so it appears in the Traces view |
| arize: prompt and response content present | Arize received `input.value` and the other content attributes |
| dynatrace: full trace (APM and AI spans) | Dynatrace stored the HTTP, database, and AI spans under the same trace id as Arize |
| dynatrace: one root (POST /ask) | Dynatrace kept the real parents: one tree, with the AI subtree under the request span |
| dynatrace: prompt and response content removed | No span in the Dynatrace copy has `input.value`, `output.value`, or `gen_ai.tool.call.arguments` |
| dynatrace: correlated logs found by trace_id | `fetch logs \| filter trace_id == "<id>"` returns the request's logs |
| dynatrace: metrics present | `example.requests` arrived as a delta metric |

Nested scenarios add a check per expected parent, for example "support_agent is a child of triage_agent". This proves that the root handling did not detach nested agents.

Collector scenarios add two checks based on the collector's self-metrics:

| Check | What it proves |
| --- | --- |
| collector: spans sent to Dynatrace, none failed | The Dynatrace exporter sent every span and no exporter reported a failure |
| collector: Arize got exactly the spans it stored | The collector sent Arize the same number of spans that Arize stored: the AI spans plus the root, nothing else |

## Manual checks

The script uses the Grail query API and the `ax` CLI. To look at a trace by hand:

- Dynatrace: open the trace id in the Distributed Tracing app, or run `fetch spans | filter trace.id == toUid("<trace id>")` in a notebook.
- Arize AX: open the project (`ax-dt-collector-example` or `ax-dt-incode-example` by default; override with `ARIZE_COLLECTOR_PROJECT` and `ARIZE_INCODE_PROJECT`) and find the trace id in the Traces view.

## Tested versions

- Collector: `dynatrace-otel-collector` 0.60.0 and `otelcol-contrib` 0.165.0, both on macOS arm64.
- Python 3.13 with `opentelemetry-sdk` 1.45.2, `openinference-instrumentation-openai` 0.1.69, and `openai` 3.48.
- Node 23 with `ai` 7.0, `@ai-sdk/otel` 1.0, `@arizeai/openinference-vercel` 3.3.1, and `@opentelemetry/sdk-trace-node` 2.12.

## Not covered

- Docker Compose and the Kubernetes manifest are not run by the script. They use the same `collector/config.yaml`.
- The tail sampling and `load_balancing` snippets in the collector guide are checked with `validate` on both distributions, not with traffic.
- Dynatrace OneAgent and this SDK setup in the same process.
