#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env.openclaw_eval}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

PYTHON_BIN="${PYTHON_BIN:-$OPENCLAW_EVAL_PYTHON}"
RUNNER="${RUNNER:-$ROOT_DIR/$OPENCLAW_EVAL_SCRIPT}"
DATASET="${DATASET:-$OPENCLAW_EVAL_DATASET}"
OUTPUT_DIR="${OUTPUT_DIR:-$OPENCLAW_EVAL_OUTPUT_DIR}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$OPENCLAW_EVAL_WORKSPACE_ROOT}"
OPENCLAW_BIN="${OPENCLAW_BIN:-$OPENCLAW_EVAL_OPENCLAW_BIN}"
AGENT="${AGENT:-$OPENCLAW_EVAL_AGENT}"
AGENT_MODEL="${AGENT_MODEL:-$OPENCLAW_EVAL_MODEL}"
START="${START:-$OPENCLAW_EVAL_START}"
END="${END:-$OPENCLAW_EVAL_END}"
PART_SIZE="${PART_SIZE:-$OPENCLAW_EVAL_PART_SIZE}"
TOP_K="${TOP_K:-$OPENCLAW_EVAL_TOP_K}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-$OPENCLAW_EVAL_REQUEST_TIMEOUT}"
DRY_RUN="${DRY_RUN:-0}"
RUN_CHECKS="${RUN_CHECKS:-1}"
CHECK_WORKSPACE="${CHECK_WORKSPACE:-Gina_1}"

if [[ "$PART_SIZE" -lt 1 ]]; then
  echo "PART_SIZE must be >= 1" >&2
  exit 2
fi

echo "Python: $PYTHON_BIN"
echo "Runner: $RUNNER"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Workspace root: $WORKSPACE_ROOT"
echo "OpenCLAW: $OPENCLAW_BIN"
echo "Agent: $AGENT"
echo "Agent model: $AGENT_MODEL"
echo "Range: $START..$((END - 1))"
echo "Part size: $PART_SIZE"
echo "Top K: $TOP_K"
echo "Request timeout: $REQUEST_TIMEOUT"
echo "Dry run: $DRY_RUN"

cd "$ROOT_DIR"

if [[ "$RUN_CHECKS" == "1" && "$DRY_RUN" != "1" ]]; then
  echo "Checking OpenCLAW agent connectivity..."
  "$OPENCLAW_BIN" agent --agent "$AGENT" --local \
    -m "reply with OK" \
    --session-id openclaw-eval-env-test

  if [[ -d "$WORKSPACE_ROOT/$CHECK_WORKSPACE" ]]; then
    echo "Checking OpenCLAW memory search in $CHECK_WORKSPACE..."
    (
      cd "$WORKSPACE_ROOT/$CHECK_WORKSPACE"
      "$OPENCLAW_BIN" memory search \
        --query "test" \
        --max-results 5 \
        --agent "$AGENT" \
        --json
    )
  else
    echo "Skipping memory search check; workspace not found: $WORKSPACE_ROOT/$CHECK_WORKSPACE"
  fi
fi

args=(
  "$RUNNER"
  --dataset "$DATASET"
  --output-dir "$OUTPUT_DIR"
  --workspace-root "$WORKSPACE_ROOT"
  --openclaw-bin "$OPENCLAW_BIN"
  --agent "$AGENT"
  --agent-model "$AGENT_MODEL"
  --start "$START"
  --end "$END"
  --part-size "$PART_SIZE"
  --top-k "$TOP_K"
  --resume
  --request-timeout "$REQUEST_TIMEOUT"
)

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON_BIN" "${args[@]}"
echo "OpenCLAW LoCoMo eval finished."
