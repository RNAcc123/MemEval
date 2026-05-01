#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/share/project/chenchen/code/MemEval"
PYTHON_BIN="${PYTHON_BIN:-/share/project/chenchen/envs/memeval/bin/python}"
RUNNER="${RUNNER:-$ROOT_DIR/scripts/run_diagnosis_longmemeval_s.py}"

INPUT_FILE="${INPUT_FILE:-$ROOT_DIR/data/input/mem0_mem/longmemeval_s/mem0_longmemeval_s_part1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/output/llm_annotation_longmemeval_s}"
MODEL="${MODEL:-deepseek}"
NUM_VOTES="${NUM_VOTES:-3}"
THREADS="${THREADS:-10}"
LIMIT="${LIMIT:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

if [[ "$THREADS" -lt 1 ]]; then
  echo "THREADS must be >= 1" >&2
  exit 2
fi

if [[ "$NUM_VOTES" -lt 1 ]]; then
  echo "NUM_VOTES must be >= 1" >&2
  exit 2
fi

args=(
  "$RUNNER"
  "$MODEL"
  --voting
  --num-votes "$NUM_VOTES"
  -i "$INPUT_FILE"
  -o "$OUTPUT_DIR"
  -t "$THREADS"
)

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

if [[ -n "$OUTPUT_FILE" ]]; then
  args+=(-f "$OUTPUT_FILE")
fi

echo "Python: $PYTHON_BIN"
echo "Runner: $RUNNER"
echo "Input: $INPUT_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Voting rounds: $NUM_VOTES"
echo "Threads: $THREADS"
if [[ -n "$LIMIT" ]]; then
  echo "Limit: $LIMIT"
fi
if [[ -n "$OUTPUT_FILE" ]]; then
  echo "Output file: $OUTPUT_FILE"
fi

cd "$ROOT_DIR/scripts"
"$PYTHON_BIN" "${args[@]}"
