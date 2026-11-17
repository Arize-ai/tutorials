# Software factory

The companion code for the Arize AX course on building a software factory that tests
its own prompts.

A software factory is a pipeline where a GitHub issue goes in and a released change
comes out, with a Codex agent doing the work at each station. This one works on
[ship-it-card](https://github.com/jimbobbennett/ship-it-card), a small Python app that
turns a merged GitHub PR into a trading card. You fork it, file issues on your fork,
and the factory takes each one from spec to release, with you as the review gate.

```
issue → spec → build → CI ⟲ babysit → 3 reviewers → router → auto-merge ─┐
                                                            └→ review gate ┴→ release
```

| Station | What it does | Sandbox |
|---|---|---|
| Spec | Turns the issue into acceptance criteria, posted as an issue comment | read-only |
| Build | Writes the code and tests in a fresh git worktree; the factory opens the PR | workspace-write |
| CI | Waits for GitHub Actions on the PR head | (no agent) |
| Babysit | Fixes red CI (up to 3 times), resolves a conflict with main, or addresses your `/changes` | workspace-write |
| Review ×3 | Correctness, security and tests reviewers run in parallel; verdicts are posted as PR reviews | read-only |
| Router | The agent picks auto-merge or the review gate; a rules floor always wins | read-only |
| Review gate | You comment `/ship` or `/changes` on the PR | (you) |
| Release | Smoke-tests the merged commit, picks the version bump and writes notes; the factory creates the GitHub release | workspace-write |

**Agents never write to GitHub.** They return structured JSON, and the factory does
every commit, push, comment, merge and release, so every write is a known, logged step.
Each agent runs in a Codex sandbox: read-only stations can't change anything, and the
others can only write inside their own worktree. Each station also has a token budget
and a time limit, so a confused agent can't run up a bill.

## Requirements

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/)
- The [Codex CLI](https://developers.openai.com/codex/cli), logged in. The factory drives
  Codex through the [Codex SDK](https://developers.openai.com/codex/sdk), which uses
  your Codex login.
- Codex tracing from [Arize Coding Harness Tracing](https://github.com/Arize-ai/coding-harness-tracing),
  installed for your Arize AX space:
  `curl -sSL https://raw.githubusercontent.com/Arize-ai/coding-harness-tracing/main/install.sh | bash -s -- codex`
- The [GitHub CLI](https://cli.github.com/) (`gh`), logged in to the account that owns
  your fork
- An [Arize AX](https://app.arize.com/auth/join) account, and the [`ax` CLI](https://arize.com/docs/ax/skills/install)
  logged in with a profile for your space (for Prompt Hub)
- Your fork of [ship-it-card](https://github.com/jimbobbennett/ship-it-card), with GitHub
  Actions enabled

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env`: `FACTORY_TARGET_REPO` (your fork), and your Arize AX space ID and API
key. Use the same space you installed Codex tracing into.

Then seed Prompt Hub with the station prompts from `factory/prompts/`:

```bash
uv run python -m factory.prompts push --promote
```

## Running the factory

File an issue on your fork, then:

```bash
uv run python -m factory.run 3            # run the factory on issue #3
```

The factory logs each station as it goes, and `runs/events.jsonl` has every event,
including each agent tool call. More ways to run it:

```bash
uv run python -m factory.run 4 5 6 -p 3   # three issues in flight at once
uv run python -m factory.run 1 --from ci  # adopt an existing factory PR, skip spec + build
uv run python -m factory.run --watch      # resume runs parked at the review gate
uv run python -m factory.run --release 7  # re-run only the release step for issue #7
```

## The review gate

When the router doesn't auto-merge, the PR gets the `needs-human` label and waits for
you. Your verdict is a PR comment that starts with:

- `/ship` to merge and release it
- `/changes <what to fix>` to send it back to the babysitter. Inline review comments you
  posted since the gate opened are passed along too.

Merging or closing the PR by hand also counts. The factory acts on GitHub as you, and
GitHub won't let you approve your own PR, which is why the verdict is a comment. Parked
runs are saved in `runs/state/`, and `--watch` picks them up and continues them in the
same Arize AX trace.

## Tracing

Two kinds of trace land in your `software-factory` project:

- **One factory trace per issue**, sent by the factory: a `factory.run` root span with a
  span per station (`station.spec`, `station.build`, `station.ci`, ...). Every span has
  `session.id` set to `<repo>#<issue>`, and station spans carry `factory.station`,
  `factory.issue`, `factory.pr` and `factory.attempt`.
- **One agent trace per Codex turn**, sent by Codex tracing when the turn completes: an
  LLM span with the prompt, the answer, the model and token counts, and a TOOL span for
  each tool call. Its `session.id` is the Codex thread ID.

The factory ties the two together. Each station span records the agent's
`factory.codex_thread_id`, and the factory passes `OTEL_RESOURCE_ATTRIBUTES` to each
agent's Codex runtime, so every agent span carries `factory.issue`, `factory.station`,
`factory.attempt`, `factory.run_trace_id` and the Prompt Hub version that ran
(`factory.prompt_name`, `factory.prompt_version_id`). To see every agent that worked on
issue 3, filter spans on `factory.issue = '3'`.

The factory sets the project for both with `FACTORY_ARIZE_PROJECT`, not
`ARIZE_PROJECT_NAME`. A traced coding-agent session (for example, Claude Code with its
own tracing) exports `ARIZE_PROJECT_NAME`, and a factory started from inside it would
otherwise send everything to that session's project.

## Prompts live in Arize Prompt Hub

Each agent station has one Prompt Hub prompt, `software-factory-<station>`, and runs the
version labeled `production`. Promoting a new version in Arize AX changes the factory's
behavior with no code change or restart. The files in `factory/prompts/` are the seed
and the fallback: if Prompt Hub can't be reached, the station uses the local text and
its spans say `prompt_source=local`.

```bash
uv run python -m factory.prompts status          # what production points at, drift vs local
uv run python -m factory.prompts push            # new versions from the local files
uv run python -m factory.prompts push --promote  # ...and move `production` to them
uv run python -m factory.prompts pull            # write production text to factory/prompts/hub/
```

## Configuration

Everything is read from `.env`:

| Variable | Default | What it does |
|---|---|---|
| `FACTORY_TARGET_REPO` | (required) | Your fork, e.g. `you/ship-it-card` |
| `ARIZE_SPACE_ID`, `ARIZE_API_KEY` | unset | Arize AX credentials; without them the factory runs untraced |
| `FACTORY_ARIZE_PROJECT` | `software-factory` | Project for the factory trace and every agent trace |
| `FACTORY_PROMPT_PREFIX` | `software-factory` | Prompt Hub names are `<prefix>-<station>` |
| `FACTORY_BUILD_MODEL` | `gpt-6.1-sol` | Codex model for spec, build, babysit, router and release |
| `FACTORY_REVIEW_MODEL` | `gpt-6-astra` | Codex model for the three reviewers |
| `FACTORY_<STATION>_MAX_TOKENS` | per station | Token budget (uncached input + output) before a turn is interrupted |
| `FACTORY_AGENT_TIMEOUT_S` | `1800` | Wall-clock limit for any one agent turn |
| `FACTORY_MAX_CI_FIXES` | `3` | Babysit attempts before a red PR is labeled `factory-stuck` |
| `FACTORY_HUMAN_WAIT_S` | `0` | How long a run waits at the review gate before handing off to `--watch` |
| `FACTORY_BIG_DIFF_LINES` | `400` | Diffs bigger than this always go to the review gate |
| `FACTORY_SENSITIVE_PATHS` | `.github/,pyproject.toml,uv.lock,shipit/github.py` | Changes here always go to the review gate |

Each agent also inherits your own Codex configuration in `~/.codex/config.toml`, which is
where Codex tracing registers its `notify` hook.

## Project layout

```
factory/
  run.py          orchestrator and CLI
  agent.py        runs one Codex turn: sandbox, strict output schema, budgets, tool ticker
  tracing.py      the factory trace, and the attributes that tie agent traces to it
  context.py      per-issue run state, saved to runs/state/
  prompts.py      Prompt Hub loader and CLI
  gh.py           git and gh plumbing (the only code that writes to GitHub)
  stations/       spec, build, ci, babysit, review, router, human, release
  prompts/        station system prompts (the Prompt Hub seed)
tests/            unit tests: uv run pytest
```
