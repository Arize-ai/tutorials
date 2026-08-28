#!/usr/bin/env bash
# Create the AX evaluator and continuous task used by this receipt tutorial.
set -euo pipefail

: "${ARIZE_SPACE_ID:?Set ARIZE_SPACE_ID to the target AX space.}"

command -v ax >/dev/null || { echo "The AX CLI must be installed." >&2; exit 1; }
command -v jq >/dev/null || { echo "jq must be installed." >&2; exit 1; }

PROJECT_NAME="${ARIZE_PROJECT_NAME:-receipt-image-evals}"
JUDGE_MODEL="${RECEIPT_JUDGE_MODEL:-gpt-5.6-luna}"
EVALUATOR_NAME="Receipt Visual Groundedness"
EVALUATOR_COLUMN="receipt_visual_groundedness"
TASK_NAME="Receipt Visual Groundedness Monitor"

read_json_id() {
  jq -er '.id'
}

find_task_id() {
  jq -er --arg name "$TASK_NAME" '
    (.tasks // .items // [])[] | select(.name == $name) | .id
  ' || true
}

if evaluator_json="$(ax evaluators get "$EVALUATOR_NAME" --space "$ARIZE_SPACE_ID" --output json 2>/dev/null)"; then
  EVALUATOR_ID="$(printf '%s' "$evaluator_json" | read_json_id)"
  echo "Reusing evaluator: $EVALUATOR_ID"
else
  : "${ARIZE_AI_INTEGRATION_ID:?Set ARIZE_AI_INTEGRATION_ID to the AX AI integration ID for the judge model.}"
  ax evaluators create-template-evaluator \
    --name "$EVALUATOR_NAME" \
    --space "$ARIZE_SPACE_ID" \
    --template-name "$EVALUATOR_COLUMN" \
    --commit-message "Create receipt visual-groundedness evaluator" \
    --ai-integration-id "$ARIZE_AI_INTEGRATION_ID" \
    --model-name "$JUDGE_MODEL" \
    --description "Checks whether receipt extraction JSON is visually grounded in its receipt image." \
    --include-explanations \
    --invocation-params '{"temperature": 0}' \
    --classification-choices '{"grounded": 1, "not_grounded": 0, "needs_review": 0}' \
    --direction MAXIMIZE \
    --data-granularity span \
    --template 'Assess whether the structured extraction is visually grounded in the receipt image.

<receipt_image>
{{receipt_image}}
</receipt_image>

<extraction>
{{extraction}}
</extraction>

Choose grounded only when every asserted merchant, item, amount, currency, and total is visibly supported. Choose not_grounded for invented or contradicted values. Choose needs_review when the image is degraded or ambiguous enough that a reliable decision cannot be made.

Return a classification label and an explanation. The explanation must be valid JSON, without Markdown fences, with exactly these fields:
{
  "evaluated_extraction": <the extraction JSON exactly as received>,
  "judge_result": {
    "label": <the selected classification label>,
    "reason": <a concise visual-grounding rationale>
  }
}'

  EVALUATOR_ID="$(ax evaluators get "$EVALUATOR_NAME" --space "$ARIZE_SPACE_ID" --output json | read_json_id)"
  echo "Created evaluator: $EVALUATOR_ID"
fi

task_json="$(ax tasks list --name "$TASK_NAME" --project "$PROJECT_NAME" --space "$ARIZE_SPACE_ID" --limit 100 --output json)"
TASK_ID="$(printf '%s' "$task_json" | find_task_id)"
if [[ -n "$TASK_ID" ]]; then
  echo "Reusing continuous evaluation task: $TASK_ID"
  exit 0
fi

EVALUATORS="$(jq -nc --arg evaluator_id "$EVALUATOR_ID" '[{
  evaluator_id: $evaluator_id,
  query_filter: "span_kind = '\''CHAIN'\''",
  column_mappings: {
    receipt_image: "attributes.input.value",
    extraction: "attributes.output.value"
  }
}]')"

ax tasks create-evaluation \
  --name "$TASK_NAME" \
  --task-type TEMPLATE_EVALUATION \
  --project "$PROJECT_NAME" \
  --space "$ARIZE_SPACE_ID" \
  --evaluators "$EVALUATORS" \
  --is-continuous \
  --sampling-rate 1

echo "Created continuous evaluation task for project: $PROJECT_NAME"
