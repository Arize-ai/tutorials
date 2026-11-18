# Maintainer notes

Working notes for whoever maintains this course: the decisions behind it, what we've measured, where it stands, and the research it draws on. If you're taking the course, start with [README.md](README.md) instead.

The course docs are in [Arize-ai/docs#1306](https://github.com/Arize-ai/docs/pull/1306). That PR has the overview and lessons 1 and 2, and its handoff comment mirrors this file.

## Decisions

- **Stack.**
  - Every station is a Codex agent, run through the [Codex SDK](https://developers.openai.com/codex/sdk).
  - [Arize Coding Harness Tracing](https://github.com/Arize-ai/coding-harness-tracing) traces each agent turn. It needs the prompt fix in [#187](https://github.com/Arize-ai/coding-harness-tracing/pull/187), or turn spans have no prompt.
  - The factory sends its own trace per issue. It also passes `factory.*` resource attributes (issue, station, attempt, run trace ID, Prompt Hub version) to every agent, so the two traces join both ways.
- **Target app: a multi-tenant helpdesk** (`app/`), chosen over the earlier ship-it-card app. Its changes aren't equally safe, which gives the router and the review gate real decisions:
  - Tenant isolation, authentication, API keys and the data model are risky.
  - A column in the ticket list is not.
  - `python -m factory.setup you/helpdesk` copies the app into each reader's own repo, so the factory has a repo to open PRs, run CI and release in.
- **Models.**
  - The defaults are `gpt-6.2-sol` for spec, build, babysit, router and release, and `gpt-6.2-astra` for the three reviewers.
  - We first ran `gpt-6.1-sol` with `gpt-6-astra`, at about $2 an issue.
  - We also tried `gpt-6-luna` with `gpt-6.2-sol`. It was about 5x cheaper but clearly worse:
    - the reviewers raised false alarms on a correct fix;
    - all three missed a CSV formula injection;
    - the builder built the cross-workspace lookup that the larger builder refused.

  Lesson 1 mentions this.
- **The backlog.** `backlog/` holds 10 issues of known difficulty and risk. Each file's front matter records the route the course expects (`expect:`) and why. Only the title and body are filed, so the agents never see the expected route.
- **Docs conventions.**
  - Diagrams follow the Arize 2026 Rebrand design system, with a light and a dark version. Their HTML sources sit next to the PNGs in the docs repo, as `images/software-factory/*.source.html`.
  - Prose follows the Arize AX docs voice. It calls the product "Arize AX", uses no dashes, and is checked against a list of AI-writing tics.

## Measured numbers

All numbers are at API prices.

| Setup | To the review gate | Release | Declined by the builder |
|---|---|---|---|
| `gpt-6.2-sol` + `gpt-6.2-astra` | ~$1.72, ~5 min | ~$0.15 | ~$0.20 |
| `gpt-6.1-sol` + `gpt-6-astra` | ~$2.10 | ~$0.13 | ~$0.19 |
| `gpt-6-luna` + `gpt-6.2-sol` | ~$0.39 | n/a | n/a |

- The three reviewers account for most of the cost in every setup.
- Arize AX doesn't price the `gpt-6.2-*` models yet, so `llm.cost.total` is `0` for those turns. We estimated costs from the token counts (uncached input, cached input and output) at list prices.
- Lesson 2 and the overview say Arize AX shows each turn's cost. Fix the pricing, or the wording, before publishing.

## Gate verdicts collected so far

These are from the author's private test repo. Lessons 3 to 5 need verdicts like these.

| PR | Issue | Route | Gate verdict | Notes |
|---|---|---|---|---|
| #3 | Reply doesn't move a ticket up | Auto-merge | none | Released v0.1.0 |
| #7 | Ticket age | Auto-merge | none | Released v0.2.0 |
| #8 | Search | Review gate | `/changes`, then `/ship` | Reviewers caught a bug (`priority=` returned 422). Released v0.4.0 |
| #9 | CSV export | Review gate (high-risk spec) | `/ship` | Formula injection handled. Released v0.3.0 |
| #16 | Pagination | Auto-merge | none | Released v0.5.0. **Missed:** offset paging over a list sorted by `updated_at` skips or repeats tickets during a sync. A good unsafe-auto-merge example for lesson 5 |
| #15 | Login lockout | Review gate | `/changes` | One valid finding (unbounded memory) and one false alarm (empty password) |
| #17 | Webhook SSRF | Review gate | `/changes` | Reviewers caught reserved IPv6 ranges |
| #18, #19 | Retries, tags | none | none | Failed before review when Codex credits ran out (`--retry` them) |

The cross-workspace lookup was blocked before build: the builder declined and handed the decision to a human.

These verdicts span three model setups. For lesson 4's numbers, collect a fresh set on one fixed setup.

## Status and next steps

1. **Finish the paused runs.** Run `factory.run --watch` for #15 and #17, then `factory.run --retry 13` and `--retry 14`.
1. **Lesson 3, "Turn gate verdicts into evals."**
   - Port the harvester from the earlier Claude-based factory, [jimbobbennett/software-factory](https://github.com/jimbobbennett/software-factory) (`factory/harvest.py`, `factory/reference.py`). It writes each gate's verdict onto the issue's trace.
   - Label each reviewer against the gate: agree, miss, or false alarm.
   - Add LLM judges that attribute each miss to one reviewer lane. Check the judges against a human review before you sample with them.
1. **Lesson 4, "Improve the reviewer."**
   - Build a dataset of PRs at the commit the reviewers saw.
   - Run experiments with 3 repetitions against a baseline.
   - Promote in Prompt Hub only when misses go down and false alarms don't go up.
   - Link each prompt version to the misses it targets.
1. **Lesson 5, "Keep the factory honest."**
   - Test the improved reviewer on new code with planted bugs.
   - Audit the gate's own verdicts.
   - Run the router loop with a code evaluator: correct, overcautious, or unsafe.
   - Use post-merge verdicts such as the pagination miss.
   - Optionally, add a model-swap experiment measured separately from prompt changes.
   - Build a health dashboard: autonomy rate, cost per issue by route, and cycle time per station.
1. **Before publishing.** Merge this code PR, merge harness tracing #187, sort out pricing for the `gpt-6.2-*` models, and write lessons 3 to 5.

## Research

The course grew out of the Arize blog post [How to build a software factory that tests its own prompts](https://arize.com/blog/how-to-build-a-software-factory-that-tests-its-own-prompts/), which used the earlier Claude-based factory. Sources that shaped it:

### Software factories in practice

- [Inside OpenAI's agentic software factory](https://newsletter.pragmaticengineer.com/p/openai-software-factory) (The Pragmatic Engineer, 2026-09-15, partly paywalled).
  - OpenAI's pipeline: outcome, Codex, CI with an agent that babysits the PR, specialist reviewer agents with risk tiers, agentic deploy, production monitoring, and incident agents.
  - High-risk changes get mandatory human review.
  - "What does observability mean in this world" is left as an open question, and this course answers it.
- [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/) (OpenAI). An agent-first codebase, with "Humans steer. Agents execute." We took its figures from secondary coverage, so check them before quoting.
- [StrongDM Software Factory](https://factory.strongdm.ai/), and [the StrongDM blog post](https://www.strongdm.com/blog/the-strongdm-software-factory-building-software-with-ai).
  - No human code review.
  - Scenarios stored outside the codebase act as a holdout set, and "satisfaction" is a probabilistic measure of success.
- [Simon Willison on StrongDM](https://simonwillison.net/2026/feb/7/software-factory/). He's skeptical of skipping human review, and notes agent-written tests only help "if they don't cheat".
- [Dan Shapiro, The Five Levels](https://www.danshapiro.com/blog/2026/01/the-five-levels-from-spicy-autocomplete-to-the-software-factory/), with [Willison's summary](https://simonwillison.net/2026/Jan/28/the-five-levels/). Levels of autonomy, up to the "dark factory".
- [Spotify, background coding agents and feedback loops](https://engineering.atspotify.com/2025/12/feedback-loops-background-coding-agents-part-3).
  - Deterministic verifiers, then an LLM judge on the diff. The judge vetoed about a quarter of sessions.
  - Early agents gamed their rewards, for example by deleting failing tests.
  - A [later InfoQ report](https://infoq.com/news/2026/03/spotify-honk-rewrite) says they removed the judge. Check that before quoting.
- [Why Ramp built Inspect](https://newsletter.pragmaticengineer.com/p/why-ramp-built-inspect) (The Pragmatic Engineer). A background agent used for code and review. The reported merge shares vary, so check them before quoting.

### Evidence on agent PRs and review

- [Chowdhury et al., arXiv:2604.03196](https://arxiv.org/abs/2604.03196): PRs reviewed only by code review agents merge much less often than PRs with human review.
- [Yoshioka et al., arXiv:2601.18749](https://www.alphaxiv.org/abs/2601.18749): review features have different effects on human PRs and agent PRs.
- Related: [arXiv:2607.21832](https://arxiv.org/pdf/2607.21832) and [arXiv:2602.02345](https://arxiv.org/html/2602.02345v1). Check which paper says what before quoting.
- Takeaway for the course: merge rate alone is a weak signal. That's why the course labels each reviewer against the gate instead.
- [Laurie Voss, AI benchmarks are breaking](https://arize.com/blog/agents-too-smart-for-benchmarks/) (Arize blog). Agents game benchmarks, which is a reason to trust traces and gate verdicts over pass rates.

### Measuring and improving a factory

- [Warp Factories, Measure & improve](https://docs.warp.dev/factories/measure-and-improve/), with [Scorers](https://docs.warp.dev/factories/measure-and-improve/scorers/), [Self-improvement](https://docs.warp.dev/factories/measure-and-improve/self-improvement/), [Benchmarks](https://docs.warp.dev/factories/benchmarks/), [Dashboard](https://docs.warp.dev/factories/factory-dashboard/), and [Testing and tuning](https://docs.warp.dev/factories/guides/testing-and-tuning-your-factory/). Ideas worth adopting:
  - An autonomy metric: merged PRs that needed no human code push.
  - Cost per PR, by PR size, and cycle time per stage.
  - Testing a judge against your own review before you sample with it.
  - A baseline, then one change at a time, with prompt and model changes tested separately, and repeated runs before you trust a difference.
  - Linking each fix to the failures it addresses.
  - Branch protection for human-only merges.

  Warp doesn't cover gate verdicts as labels, per-reviewer misses, or a risk router. That's what this course adds.
- [Self-improving agents talk](https://github.com/jimbobbennett/self-improving-agents-talk). A Codex SDK improver loop:
  - a baseline experiment and 3 candidate fixes, each in its own worktree and experiment;
  - deterministic gates on each evaluator, on cost and on latency;
  - a human-reviewed PR at the end;
  - "evals are your unit tests".

  It's a model for lesson 4's promotion gate.
- Arize AX docs: [Tracing coding agents](https://arize.com/docs/ax/cookbooks/instrument/tracing-coding-agents), [Observe and optimize coding agent workflows](https://arize.com/docs/ax/cookbooks/improve/observe-and-optimize-coding-agent-workflows), and [Codex tracing](https://arize.com/docs/ax/integrations/platforms/codex/codex-tracing). None of them link PR, merge or review outcomes back to agent sessions, which is the gap this course fills.
