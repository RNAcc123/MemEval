#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QA_THREADS="${QA_THREADS:-4}"

"${PYTHON_BIN:-python}" scripts/run_diagnosis.py deepseek \
  --no-voting \
  --qa-threads "${QA_THREADS}" \
  -t 1 \
  -i "$ROOT_DIR/data/input/mem0_mem/locomo10/qwen3.5-35b-a3b/mem0_locomo10_part1.json" \
  -o "$ROOT_DIR/data/output/llm_annotation_single" \
  -f mem0_locomo10_part1_qwen3.5-35b-a3b_single_deepseek.json
