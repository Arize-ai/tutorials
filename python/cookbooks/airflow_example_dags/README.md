# Arize AX Airflow Example DAGs

A collection of 22 example DAGs demonstrating the [`arize-ax-airflow-provider`](https://github.com/Arize-ai/arize-ax-airflow) — the official Apache Airflow provider for [Arize AX](https://arize.com/docs/ax/).

Each DAG illustrates a real LLMOps workflow you can adapt for your own pipelines: CI/CD evaluation gates, prompt lifecycle management, drift detection with auto-rollback, RAG evaluation, dataset curation from production traces, fine-tuning data pipelines, behavioral regression checks, evaluator calibration, self-optimizing prompt loops, cloud-storage round-trips, and more.

> **Last Updated:** 2026-05-21

---

## Prerequisites

- **Python** 3.10+
- **Apache Airflow** 2.4+ (works on Airflow 3.x as well — DAGs use the modern `schedule=` and `from __future__ import annotations` style)
- An **Arize AX account** with an API key — sign up at [arize.com](https://arize.com/) and grab a key from **Settings → API Keys**

---

## Install

```bash
pip install arize-ax-airflow-provider
```

Some DAGs need extra packages on the worker:

| DAG | What to install | Why |
|-----|-----------------|-----|
| `example_arize_ax_prompt_optimization_with_feedback_dag.py`, `example_arize_ax_self_optimizing_loop_dag.py` | `prompt-learning-enhanced` (from git) + `arize-phoenix-evals<3.0` | Prompt Learning SDK for meta-prompt optimization; installed directly from git because PyPI rejects direct-URL deps |
| `example_arize_ax_admin_dag.py` (prompt tasks) | `arize[PromptHub]` | Requires the Arize `[PromptHub]` extra |

```bash
# Prompt Learning SDK (used by prompt_optimization_with_feedback and self_optimizing_loop)
pip install 'arize-phoenix-evals>=2.0,<3.0' \
            'prompt-learning-enhanced @ git+https://github.com/Arize-ai/prompt-learning.git'

# PromptHub extra (used by admin DAG prompt tasks)
pip install 'arize[PromptHub]'
```

---

## Set up the Airflow connection

The DAGs all use a single connection named **`arize_ax_default`**.

**Airflow UI** → **Admin → Connections → Add**:

| Field | Value |
|---|---|
| Connection Id | `arize_ax_default` |
| Connection Type | `Arize AX` |
| Password | _your Arize API key_ |
| Extra (optional) | `{"space_id": "U3BhY2U6...", "region": "us"}` |

Or via the CLI:

```bash
airflow connections add arize_ax_default \
    --conn-type arize_ax \
    --conn-password "$ARIZE_AX_API_KEY" \
    --conn-extra '{"space_id": "U3BhY2U6..."}'
```

If you set `space_id` in **Extra**, you can omit `space_id=` on individual operators and skip the `arize_ax_space_id` Variable below.

---

## Configure Airflow Variables

The DAGs read configuration from Airflow Variables so you can run them without editing source. **Admin → Variables** (or `airflow variables set <key> <value>`):

### Common (used by most DAGs)

Most DAGs run their evaluation tasks via **Arize Eval Hub** (server-side execution) rather than calling LLMs from Airflow workers. That means you'll typically need a configured AI Integration in Arize and a project to scope the eval tasks to:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `arize_ax_space_id` | If not set in connection Extra | — | Target Arize space (global ID or name) |
| `arize_ai_integration_id` (or env var `ARIZE_AI_INTEGRATION_ID`) | ✅ for any DAG running Eval Hub tasks | — | UUID of an AI Integration in Arize (Settings → AI Integrations) configured with a real provider API key. Used by `experiment_dag`, `llm_cicd_gate_dag`, `llm_experiments_dag`, `prompt_lifecycle_dag`, `drift_detection_dag`, `rag_evaluation_dag`, `self_optimizing_loop_dag`, `tasks_dag`, `evaluators_dag` |
| `arize_ax_project_id` | ✅ for any DAG running Eval Hub tasks | — | Project ID used to scope evaluation tasks. Required by the same DAGs as `arize_ai_integration_id` above |

### Per-DAG

| DAG | Variable | Required | Purpose |
|---|---|---|---|
| `annotation_queues_dag` | `arize_annotator_email` | ✅ | Comma-separated reviewer email(s), e.g. `alice@example.com,bob@example.com` |
| `annotation_queues_dag` | `arize_queue_name` | optional | Demo queue name (default: `airflow-demo-queue`) |
| `dataset_curation_dag` | `arize_ax_dataset_id` | ✅ | Target dataset to append curated examples to |
| `prompt_ab_test_dag` | `arize_ax_prompt_names` | optional | JSON list or CSV of prompt names to compare |
| `prompt_optimization_with_feedback_dag` | `arize_ax_lookback_days` | optional | Days of production feedback to learn from |
| `tasks_dag` / `evaluators_dag` | `ARIZE_EVALUATOR_MODEL` | ✅ | Model name used by the LLM-as-judge (e.g. `gpt-4o`, `claude-sonnet-4-5`) |
| `e2e_dag` | `arize_annotator_email` | optional | Enables the annotation-queue lifecycle phase |
| `self_optimizing_loop_dag` | `arize_ax_self_optimizing_model` | optional | OpenAI model used by experiment tasks (default `gpt-4o-mini`) |
| `self_optimizing_loop_dag` | `arize_ax_self_optimizing_cleanup` | optional | Set to `"true"` to delete the demo dataset on DAG completion (default `"false"`) |
| `cloud_export_dag` | `arize_ax_project_name` | ✅ | Name of the Arize project to export spans from (project name, not base64 ID) |
| `cloud_export_dag` | `arize_ax_cloud_target_prefix` | ✅ | Scheme + bucket / container, e.g. `s3://demo-bucket`, `gs://demo-bucket`, `abfs://demo-container` |
| `cloud_export_dag` | `arize_ax_cloud_storage_options` | ✅ | JSON dict of credentials / endpoint overrides for the target backend (AWS keys, MinIO endpoint, GCS service-account path, Azure account key) |

The required values are documented in each DAG's module docstring — open the file and check the `**Required**` / `**Optional Airflow Variables**` sections.

---

## Drop the DAGs into your Airflow instance

Copy the files into your Airflow `dags/` folder:

```bash
cp python/cookbooks/airflow_example_dags/*.py $AIRFLOW_HOME/dags/
```

Or symlink the directory so updates propagate:

```bash
ln -s "$(pwd)/python/cookbooks/airflow_example_dags" $AIRFLOW_HOME/dags/arize_ax_examples
```

Refresh the Airflow UI — all 22 DAGs should appear under the `arize_ax` tag.

---

## Run your first DAG

Start with a small, self-contained one to verify your setup:

```bash
airflow dags trigger example_arize_ax_dataset
```

This DAG lists datasets, creates a tiny demo dataset, and fetches it back — a 10-second round-trip that confirms the connection, API key, and space are wired correctly.

From there, work up to the workflow that matches your use case. The table below maps patterns to DAGs:

| Pattern | DAG |
|---|---|
| Smoke test (datasets) | `example_arize_ax_dataset_dag.py` |
| Run an experiment with a task + evaluator | `example_arize_ax_experiment_dag.py` |
| LLM CI/CD gate (fail on regression) | `example_arize_ax_llm_cicd_gate_dag.py` |
| Prompt lifecycle: staging → production | `example_arize_ax_prompt_lifecycle_dag.py` |
| Prompt A/B testing | `example_arize_ax_prompt_ab_test_dag.py` |
| Drift detection with auto-rollback | `example_arize_ax_drift_detection_dag.py` |
| Behavioral regression detection | `example_arize_ax_behavioral_regression_dag.py` |
| Evaluator calibration vs human labels | `example_arize_ax_evaluator_calibration_dag.py` |
| RAG evaluation pipeline | `example_arize_ax_rag_evaluation_dag.py` |
| Curate production spans into a dataset | `example_arize_ax_dataset_curation_dag.py` |
| Fine-tuning data pipeline | `example_arize_ax_finetune_data_pipeline_dag.py` |
| Continuous evaluation tasks (Eval Hub) | `example_arize_ax_tasks_dag.py` |
| Human-in-the-loop annotation queues | `example_arize_ax_annotation_queues_dag.py` |
| Multi-model experiment matrix | `example_arize_ax_llm_experiments_dag.py` |
| Self-learning prompt optimization | `example_arize_ax_prompt_optimization_with_feedback_dag.py` |
| Self-optimizing prompt loop (baseline → optimize → gate → promote) | `example_arize_ax_self_optimizing_loop_dag.py` |
| End-to-end provider smoke test (~70 operators, 7 sensors) | `example_arize_ax_e2e_dag.py` |
| Cloud-storage round-trip (spans → S3/GCS/ABFS → dataset) | `example_arize_ax_cloud_export_dag.py` |
| Inventory / admin (list spaces, projects) | `example_arize_ax_admin_dag.py` |
| Span export & metrics | `example_arize_ax_spans_dag.py` |
| Custom evaluator creation | `example_arize_ax_evaluators_dag.py` |
| ML (batch logging) | `example_arize_ax_ml_dag.py` |

---

## Adapting a DAG for your environment

These are demo DAGs — `schedule=None` and `catchup=False` so they only run on manual trigger. To productionize one:

1. **Set a schedule** — `schedule="@daily"` or a cron expression.
2. **Replace demo seeds** — the small inline example lists (e.g. `{"query": "What is 2+2?", ...}`) are placeholders; point the DAG at your real dataset by setting the appropriate Variable.
3. **Tighten cleanup tasks** — most demos delete the resources they create (via `trigger_rule="all_done"` cleanup tasks). Remove those if you want the resources to persist across runs.
4. **Add alerting** — wire `on_failure_callback` / `on_retry_callback` to your Slack/PagerDuty hook for the gate DAGs (`llm_cicd_gate`, `drift_detection`, `behavioral_regression`).

---

## Further reading

- **Provider guide:** [arize.com/docs/ax/integrations/orchestration/airflow/airflow-provider](https://arize.com/docs/ax/integrations/orchestration/airflow/airflow-provider)
- **Operator reference:** [arize.com/docs/ax/integrations/orchestration/airflow/airflow-operators](https://arize.com/docs/ax/integrations/orchestration/airflow/airflow-operators)
- **Arize AX docs:** [arize.com/docs/ax](https://arize.com/docs/ax/)

---

## 💬 Questions or feedback?

Reach out via the **[Arize community Slack](https://arize-ai.slack.com/)** or [support](https://arize.com/support/).
