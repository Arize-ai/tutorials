---
name: check-models
description: Find and update out-of-date OpenAI and Anthropic model references (in docs, MDX, notebooks, and code) to the latest size-equivalent models, and apply the code changes each new generation requires (e.g. max_tokens → max_completion_tokens for GPT-5). Use when asked to "check the models", "update model versions", "are these models current", "migrate the models in the docs/tutorials", or before publishing content that names a model.
---

# check-models

Keep model references current. OpenAI and Anthropic ship new generations often, and docs/tutorials drift: a notebook pinned to `gpt-4o-mini` or `claude-3-5-sonnet` is teaching a model that's a generation (or three) behind, sometimes with code that no longer runs (GPT-5 rejects `max_tokens`).

This skill does three things:

1. **Look up** the latest models (don't trust memory — models change).
2. **Migrate** every reference to the latest model **of the same size tier** (a `*-mini` becomes the latest mini, never the flagship).
3. **Fix the code** each new generation requires (parameter renames/removals).

## The golden rule: match the size tier

Never "upgrade" a small model to a big one. `gpt-4o-mini` is a small/cheap model — its modern equivalent is the latest **mini**, not the flagship. Migrating it to `gpt-5.5` silently multiplies a reader's cost. Map within the tier:

| Old tier | OpenAI → | Anthropic → |
|---|---|---|
| flagship / full | latest flagship | latest Opus |
| pro | latest pro | latest Opus |
| mini / small / cheap | latest **mini** | latest Haiku |
| nano | latest **nano** | (smallest tier) latest Haiku |

> The latest mini/nano are **not** always in the newest generation. As of the policy date below, the flagship is GPT-5.5 but there is no GPT-5.5-mini — so `gpt-4o-mini` → `gpt-5.4-mini`, **not** `gpt-5.5-mini`. Always confirm which generation actually has a mini/nano variant.

## Current models (policy)

The authoritative values live in [`models.json`](./models.json) — the scanner and the GitHub Action both read it. As of `updated: 2026-06-17`:

**OpenAI (reasoning — default targets)** — flagship `gpt-5.5`, pro `gpt-5.5-pro`, mini `gpt-5.4-mini`, nano `gpt-5.4-nano`.
**OpenAI (non-reasoning — for temperature-dependent code)** — flagship `gpt-4.1`, mini `gpt-4.1-mini`, nano `gpt-4.1-nano`. Active, **not** deprecated, and they **accept `temperature`/`top_p`**.
**Anthropic** — Opus `claude-opus-4-8`, Sonnet `claude-sonnet-4-6`, Haiku `claude-haiku-4-5`.

> ⚠️ **Claude Fable 5 / Mythos 5** were export-control-suspended on 2026-06-12 — they are NOT valid migration targets. ⚠️ GPT-5.5 has **no** mini/nano variant.

**Specialised (non-chat) OpenAI models** — realtime, audio, image, transcribe, tts, embeddings, and moderation models don't map to the flagship/mini tiers, so they're tracked separately in `models.json` → `specialized`:
- `specialized.current` (e.g. `gpt-realtime-1.5`, `gpt-realtime-2`, `gpt-audio-1.5`, `gpt-image-2`, `gpt-4o-transcribe`, `gpt-4o-mini-tts`, `text-embedding-3-*`, `omni-moderation`) are **valid** — the scanner leaves them alone.
- `specialized.deprecated` maps a retired ID to its current replacement (e.g. `gpt-4o-realtime-preview` → `gpt-realtime-1.5` (shut down 2026-05-07), `gpt-4o-audio-preview` → `gpt-audio-1.5`, `gpt-4o-search-preview` → `gpt-5.4-mini`) — flagged `warn` with the target, since the realtime/audio API surface differs and the swap needs a human eye.

These are checked **before** `review.openaiPatterns`, which stays the fallback for any unrecognised variant. When a new specialised model ships (or one is retired), update these two lists.

### Reasoning vs non-reasoning — which OpenAI target?

The GPT-5 family are **reasoning** models: they reject `temperature`, `top_p`, and the other sampling params. So the target depends on whether the call relies on those:

- **General code** (no meaningful `temperature`, or `temperature` was incidental) → migrate to the **GPT-5** tier above (the default). This is what most code wants.
- **Temperature-dependent code** → migrate to the **non-reasoning `gpt-4.1` tier** and **keep `temperature`**. This is the right call for:
  - **LLM-as-judge / evaluators** that set `temperature=0` for reproducible scores (e.g. Phoenix `OpenAIModel(...)`, `LLM(...)`, eval classifiers),
  - **deterministic extraction / classification** pinned at `temperature=0`.
  Match the size tier: a `gpt-4o-mini` eval judge → `gpt-4.1-mini` (keep `temperature=0`), not `gpt-5.4-mini`.

`gpt-4.1` / `gpt-4.1-mini` / `gpt-4.1-nano` are treated as **current** by the scanner (not flagged) — they're a legitimate, non-deprecated choice. `gpt-4o` / `gpt-4o-mini` are also non-reasoning + active but older; migrate them to the GPT-5 tier by default, or to `gpt-4.1*` when temperature matters.

## Workflow

### 1. Verify the latest models are still current

`models.json` is a cache — refresh it before a big migration. **The scanner can't browse, so the lookup is yours to run; `--refresh` gives you the exact checklist:**

```bash
node .agents/skills/check-models/scripts/scan-models.mjs --refresh
```

It prints the current policy, the WebSearch queries, the authoritative source URLs, and which keys to edit. Then:

- WebSearch `"latest OpenAI models"` and `"latest Anthropic Claude models"` (use the current month); cross-check Anthropic against the **`claude-api`** skill's `shared/models.md` (canonical Claude IDs + the Claude-side code changes).
- Confirm each tier's latest model **and that the tier still exists** (a new flagship may ship with no `-mini` yet — keep mini/nano on the older generation). Skip any suspended/withdrawn model.
- If anything changed, **propose the `models.json` diff and confirm before writing** (bump `flagship`/`mini`/`opus`/… and `updated`). That one edit updates the scanner, the skill, and the CI gate together.

**When to refresh:** the scanner tells you. Every run prints a `⚠ Model policy may be out of date…` hint (and sets `stale.suggestRefresh` in `--json`) when either the policy is older than ~45 days **or** the content references a model newer than the policy knows (e.g. a `gpt-5.6` appears while the policy flagship is `gpt-5.5`). Newer-than-policy models are left untouched — never downgraded — so refresh first, then re-scan.

### 2. Scan

```bash
node .agents/skills/check-models/scripts/scan-models.mjs python typescript   # scan paths
node .agents/skills/check-models/scripts/scan-models.mjs --json > out.json   # machine-readable
```

Each finding is one of:
- `✗ error` — an outdated lowercase canonical model ID (e.g. `gpt-4o-mini`). Migrate it.
- `⚠ review` / `⚠ replace` (prose) — a model named in prose (`GPT-4o`) or a specialised variant (`*-codex`, `*-chat-latest`, `gpt-4v`). Use judgement (see below).
- `⚠ param` — a GPT-5/o-series code change to apply (`max_tokens`, `temperature`, …).

### 3. Migrate the model IDs

For each finding, replace with the scanner's suggested target, **preserving local style**:
- Keep the original separator/case for prose: `GPT-4o` → `GPT-5.5` (not `gpt-5.5`); `claude-sonnet-4.5` (dotted) → `claude-sonnet-4.6`.
- Drop stale date snapshots: `gpt-5-2025-08-07` → `gpt-5.5` (use the bare alias, don't invent a date).
- Keep both halves of any SDK-version tab in sync (v7/v8 examples).

#### Platform-specific IDs (Bedrock, Databricks, OpenRouter, LiteLLM)

The scanner matches the embedded `claude-*` / `gpt-*` substring inside a platform-wrapped ID and flags it. Bump the **version** but **keep the host platform's ID format** — these are not bare first-party IDs:

- **Amazon Bedrock** — IDs look like `[region.]anthropic.claude-<...>[-vN:0]`. Claude 4.x (Opus 4.x, Sonnet 4.5+, Haiku 4.5) require a **cross-region inference profile**, so they take a `us.` / `eu.` / `apac.` prefix and **drop** the on-demand `-vN:0` suffix the Claude 3 IDs used. e.g. `anthropic.claude-3-haiku-20240307-v1:0` → `us.anthropic.claude-haiku-4-5`. Match the region prefix to the doc's endpoint (the repo's Bedrock docs default to `us.`).
- **Databricks** — Foundation Model endpoint names look like `databricks-claude-sonnet-4-6`. Databricks owns these names and **availability is workspace/region-dependent** — bump the version following their pattern, but verify the endpoint actually exists rather than assuming it.
- **OpenRouter / LiteLLM** — provider-prefixed: `anthropic/claude-sonnet-4-6`, `openai/gpt-5.4-mini`. Keep the `provider/` prefix; bump only the model half.

When a migration changes more than the version number (a Bedrock region prefix, a Databricks endpoint name), **call it out for the reviewer** — it may need adjusting for their region/workspace.

### 4. Apply the code changes

**These changes apply to the _raw OpenAI SDK_ only** (`client.chat.completions.create(...)`, `openai.OpenAI()...`). When migrating such a call **to GPT-5 or o-series** (reasoning models), in the same example:
- Rename `max_tokens` → `max_completion_tokens`.
- Remove `temperature` (unless it's the default `1`), `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `logit_bias` — reasoning models reject them. Steer with `reasoning_effort` (`low`/`medium`/`high`) and `verbosity` instead.

**Wrapper libraries need a split rule** — `phoenix.evals.OpenAIModel(...)`, `langchain_openai.ChatOpenAI(...)`, `litellm.completion(...)`, etc. expose their own kwargs:
- **`max_tokens`: keep it.** The wrapper owns this kwarg and maps it to `max_completion_tokens` internally; renaming it passes an unknown constructor arg and breaks the call.
- **non-default `temperature` / `top_p`:** a GPT-5/o-series reasoning model rejects any `temperature` ≠ 1 with a 400 — wrapper or not. Two correct outcomes:
  - if the value **matters** (eval judge / deterministic call) → re-target to the non-reasoning **`gpt-4.1`** tier and **keep** the `temperature` (see "Reasoning vs non-reasoning" above);
  - if it was **incidental** → drop the explicit value and stay on the GPT-5 model.
- Otherwise migrate the **model ID only**.

> **`phoenix.evals` gotcha:** the legacy model classes (`OpenAIModel`, `LiteLLMModel`, `AnthropicModel`) accept `temperature` in their constructor (valid). The newer `LLM(provider=…, model=…)` does **not** — its `**kwargs` are forwarded to the *SDK client constructor* (for `api_key`/`base_url`), so `LLM(provider="openai", model="gpt-4.1", temperature=0)` raises `TypeError` at construction. Set sampling params on the **evaluator** instead: `ClassificationEvaluator(name=…, llm=…, prompt_template=…, choices=…, temperature=0)` (the `create_classifier(...)` helper takes no `temperature`).

(The scanner flags any `max_tokens`/`temperature` token as a `⚠ param` — it can't tell a raw call from a wrapper or know the value, so this is the judgement call.)

**Watch token caps on reasoning models.** GPT-5/o-series count *reasoning* tokens against `max_completion_tokens`, so a tiny cap that worked on gpt-4o (e.g. `max_tokens=20` for a short answer) can return empty/truncated output. When migrating such a call, raise the cap to a safe value (≥256) or set `reasoning_effort: "minimal"`.

When migrating **to Claude Opus 4.8/4.7 or Sonnet 4.6**: `budget_tokens` and `temperature`/`top_p`/`top_k` are removed — use `thinking: {type: "adaptive"}` + `output_config.effort`. Defer to the **`claude-api`** skill (`/claude-api migrate`) for the full Claude code-change checklist; don't hand-edit Claude SDK calls from memory.

### 5. Skip what shouldn't change

Do **not** rewrite:
- **Autogenerated code** — `api-clients/` is generated SDK reference; never hand-edit it (it's in `excludePaths`, so the scanner skips it). Fix the model references at their generator/source instead.
- **Historical / release-notes / changelog / migration-guide** content — it documents what *was* true ("v1.2 added gpt-4o support"). These paths are in `excludePaths`; respect the same rule for any historical prose the scanner happens to catch.
- **Non-model tokens** that share a prefix: `gpt-oss-*` (open-weight models, version-pinned), `claude-code`, `claude-agent-sdk`, web-crawler user-agents (`claude-web`, `claude-user`, `claude-searchbot`), image filenames. These are in the `ignore` list and won't be flagged.
- **Comparative prose** that names an old model on purpose ("unlike GPT-4, GPT-5.5 can…"). Leave the historical reference; only update where the doc is telling the reader which model to *use*.
- **Markdown image alt text** — `![…model gpt-4o-mini…](screenshot.png)` describes what a screenshot *shows*. Editing the alt text alone would make it misdescribe the image (the pixels still show the old model), so the scanner skips model names inside `![ … ]`. Update these by **regenerating the screenshot**, not by editing the alt text.

To suppress a single line the scanner shouldn't touch, add a `check-models:ignore` comment on it.

### 6. Verify

Re-run the scanner — `✗ error` count should be 0. Remaining `⚠` are prose/variants you've consciously reviewed.

Also guard against a **stale date carried onto a new alias**: when migrating a dated ID, drop the date (`claude-sonnet-4-5-20250929` → `claude-sonnet-4-6`, **not** `claude-sonnet-4-6-20250929` — that snapshot doesn't exist). The scanner can't catch this (it date-strips before classifying, so a wrong-dated current alias looks current), so grep for it:

```bash
grep -rnE 'claude-(opus-4-8|sonnet-4-6)-20[0-9]' python typescript   # these aliases have no dated form
```

## Updating for a new model launch

The whole skill is driven by `models.json`. When a new model ships: WebSearch to confirm the IDs and which tiers exist, edit the relevant value(s) in `models.json` plus `updated`, re-run the scanner. No code changes to the scanner are normally needed — it derives the full old→new table from the policy numbers.
