#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${RUNNER:-$ROOT_DIR/eval/longmemeval-memory/run_longmemeval_mem0_trace.py}"

DATASET="${DATASET:-$ROOT_DIR/data/input/longmemeval/longmemeval_s_cleaned.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/input/mem0_mem/longmemeval_s}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
MEM0_REPO="${MEM0_REPO:-$ROOT_DIR/../mem0}"
STORE_BASE="${STORE_BASE:-$OUTPUT_DIR/local_mem0_workers}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

START="${START:-0}"
END="${END:-500}"
PART_SIZE="${PART_SIZE:-100}"
TOP_K="${TOP_K:-30}"
CONCURRENCY="${CONCURRENCY:-2}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
MODEL="${MODEL:-env:MODEL}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-env:EMBEDDING_MODEL}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_DIR" "$STORE_BASE" "$LOG_DIR"

if [[ "$CONCURRENCY" -lt 1 ]]; then
  echo "CONCURRENCY must be >= 1" >&2
  exit 2
fi

if [[ "$PART_SIZE" -lt 1 ]]; then
  echo "PART_SIZE must be >= 1" >&2
  exit 2
fi

echo "Python: $PYTHON_BIN"
echo "Runner: $RUNNER"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Range: $START..$((END - 1))"
echo "Part size: $PART_SIZE"
echo "Concurrency: $CONCURRENCY"
echo "Dry run: $DRY_RUN"
echo "Logs: $LOG_DIR"

running=0
job_id=0

for ((part_start = START; part_start < END; part_start += PART_SIZE)); do
  part_end=$((part_start + PART_SIZE))
  if [[ "$part_end" -gt "$END" ]]; then
    part_end="$END"
  fi

  job_id=$((job_id + 1))
  worker_store="$STORE_BASE/worker_${job_id}"
  log_file="$LOG_DIR/part_${part_start}_${part_end}.log"

  echo "Launching job $job_id: samples $part_start..$((part_end - 1)) -> $log_file"
  (
    cd "$ROOT_DIR"
    export MEM0_DIR="$worker_store/mem0_home"
    args=(
      "$RUNNER"
      --dataset "$DATASET"
      --output-dir "$OUTPUT_DIR"
      --env-file "$ENV_FILE"
      --mem0-repo "$MEM0_REPO"
      --mem0-store-dir "$worker_store"
      --start "$part_start"
      --end "$part_end"
      --part-size "$PART_SIZE"
      --top-k "$TOP_K"
      --model "$MODEL"
      --embedding-model "$EMBEDDING_MODEL"
      --sleep-seconds "$SLEEP_SECONDS"
      --request-timeout "$REQUEST_TIMEOUT"
      --resume
    )
    if [[ "$DRY_RUN" == "1" ]]; then
      args+=(--dry-run)
    fi
    "$PYTHON_BIN" "${args[@]}"
  ) >"$log_file" 2>&1 &

  running=$((running + 1))
  if [[ "$running" -ge "$CONCURRENCY" ]]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "All LongMemEval memory jobs finished."
