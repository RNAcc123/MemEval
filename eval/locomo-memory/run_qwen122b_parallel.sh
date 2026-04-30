#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/share/project/chenchen/code/MemEval"
PYTHON="${PYTHON:-/share/project/chenchen/envs/memeval/bin/python}"
RUNNER="${ROOT_DIR}/eval/locomo-memory/run_locomo_mem0_trace.py"

MODEL="qwen3.5-122b-a10b"
BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
API_KEY_ENV="${API_KEY_ENV:-DASHSCOPE_API_KEY}"

START="${START:-0}"
END="${END:-10}"
TOTAL="${TOTAL:-$((END - START))}"
WORKERS="${WORKERS:-5}"
PART_SIZE="${PART_SIZE:-10}"
TOP_K="${TOP_K:-30}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/input/mem0_mem/locomo10/${MODEL}}"
STORE_ROOT="${STORE_ROOT:-${ROOT_DIR}/data/input/mem0_mem/locomo10/local_mem0_${MODEL}_workers}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

mkdir -p "${OUTPUT_DIR}" "${STORE_ROOT}" "${LOG_DIR}"

cd "${ROOT_DIR}"

if (( WORKERS <= 0 )); then
  echo "WORKERS must be positive" >&2
  exit 1
fi

if (( END <= START )); then
  echo "END must be greater than START" >&2
  exit 1
fi

TOTAL=$((END - START))
CHUNK=$(((TOTAL + WORKERS - 1) / WORKERS))

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for worker in $(seq 0 $((WORKERS - 1))); do
  worker_start=$((START + worker * CHUNK))
  worker_end=$((worker_start + CHUNK))
  if (( worker_start >= END )); then
    continue
  fi
  if (( worker_end > END )); then
    worker_end="${END}"
  fi

  store_dir="${STORE_ROOT}/worker_${worker_start}_${worker_end}"
  log_file="${LOG_DIR}/worker_${worker_start}_${worker_end}.log"
  mkdir -p "${store_dir}"

  echo "Starting ${MODEL} worker ${worker}: ${worker_start}-${worker_end}"
  MEM0_DIR="${store_dir}/mem0_home" "${PYTHON}" "${RUNNER}" \
    --model "${MODEL}" \
    --llm-api-key-env "${API_KEY_ENV}" \
    --llm-base-url "${BASE_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --mem0-store-dir "${store_dir}" \
    --start "${worker_start}" \
    --end "${worker_end}" \
    --part-size "${PART_SIZE}" \
    --top-k "${TOP_K}" \
    --request-timeout "${REQUEST_TIMEOUT}" \
    --sleep-seconds "${SLEEP_SECONDS}" \
    --resume \
    >"${log_file}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "One or more ${MODEL} workers failed. Logs: ${LOG_DIR}" >&2
  exit "${status}"
fi

echo "All ${MODEL} workers completed. Output: ${OUTPUT_DIR}"
