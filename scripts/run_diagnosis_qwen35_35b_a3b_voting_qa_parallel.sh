#!/usr/bin/env bash
set -euo pipefail

cd /share/project/chenchen/code/MemEval

QA_THREADS="${QA_THREADS:-4}"

/share/project/chenchen/envs/memeval/bin/python scripts/run_diagnosis.py deepseek \
  --voting \
  --num-votes 3 \
  --qa-threads "${QA_THREADS}" \
  -t 1 \
  -i /share/project/chenchen/code/MemEval/data/input/mem0_mem/locomo10/qwen3.5-35b-a3b/mem0_locomo10_part1.json \
  -o /share/project/chenchen/code/MemEval/data/output/llm_annotation_voting \
  -f mem0_locomo10_part1_qwen3.5-35b-a3b_voting_3rounds_deepseek.json
