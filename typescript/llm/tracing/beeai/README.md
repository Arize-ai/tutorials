This is a tutorial for tracing LLM calls using BeeAI. You'll need to have Ollama installed along with a Phoenix API key.

[Signup for Phoenix Cloud here](https://app.phoenix.arize.com).
[Install Ollama with llama 3.2 here](https://ollama.com).

## Setup

This project requires Node.js v18 or later. We recommend using Node Version Manager (nvm) to manage your Node.js versions.

```bash
nvm use 20
yarn install
```

## Configure tracing

Set your Phoenix credentials before running. For Phoenix Cloud, find your workspace URL on your Phoenix dashboard (e.g. `https://app.phoenix.arize.com/s/<your-workspace>`) and your API key under **Settings → API Keys**:

```bash
export PHOENIX_API_KEY="<your-phoenix-api-key>"
export PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com/s/<your-workspace>"
```

If `PHOENIX_COLLECTOR_ENDPOINT` is not set, traces are sent to `http://localhost:6006` (the default for a locally-running Phoenix instance).

## Run

```bash
yarn start
```