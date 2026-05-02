#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/share/project/chenchen/code/MemEval"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

PYTHON="${PYTHON:-/share/project/chenchen/envs/memeval/bin/python}"
RUNNER="${ROOT_DIR}/eval/openclaw-memory/run_locomo_openclaw_trace.py"

START="${START:-0}"
END="${END:-10}"
WORKERS="${WORKERS:-4}"
PART_SIZE="${PART_SIZE:-10}"
TOP_K="${TOP_K:-30}"
QA_WORKERS="${QA_WORKERS:-1}"
PHASE="${PHASE:-all}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"
AGENT="${AGENT:-main}"
AGENT_MODEL="${AGENT_MODEL:-}"
FAIL_FAST="${FAIL_FAST:-1}"
PREFLIGHT="${PREFLIGHT:-1}"
CLEAN_PLUGIN_CACHE="${CLEAN_PLUGIN_CACHE:-0}"
RESUME="${RESUME:-1}"
RUN_ID="${RUN_ID:-locomo-memory-$(date +%Y%m%d-%H%M%S)-$$}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/input/openclaw_mem/locomo10}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${OUTPUT_DIR}/workspaces}"
MEMORY_STATE_DIR="${MEMORY_STATE_DIR:-${OUTPUT_DIR}/memory_states}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
PREFLIGHT_WORKSPACE="${PREFLIGHT_WORKSPACE:-${OUTPUT_DIR}/preflight_workspace}"

mkdir -p "${OUTPUT_DIR}" "${WORKSPACE_ROOT}" "${MEMORY_STATE_DIR}" "${LOG_DIR}"

cd "${ROOT_DIR}"

if (( WORKERS <= 0 )); then
  echo "WORKERS must be positive" >&2
  exit 1
fi

if (( END <= START )); then
  echo "END must be greater than START" >&2
  exit 1
fi

if [[ "${FAIL_FAST}" != "0" && "${FAIL_FAST}" != "1" ]]; then
  echo "FAIL_FAST must be 0 or 1" >&2
  exit 1
fi

if [[ "${PREFLIGHT}" != "0" && "${PREFLIGHT}" != "1" ]]; then
  echo "PREFLIGHT must be 0 or 1" >&2
  exit 1
fi

if [[ "${CLEAN_PLUGIN_CACHE}" != "0" && "${CLEAN_PLUGIN_CACHE}" != "1" ]]; then
  echo "CLEAN_PLUGIN_CACHE must be 0 or 1" >&2
  exit 1
fi

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 1
fi

if [[ "${PHASE}" != "all" && "${PHASE}" != "memory" && "${PHASE}" != "answer" ]]; then
  echo "PHASE must be all, memory, or answer" >&2
  exit 1
fi

if [[ "${CLEAN_PLUGIN_CACHE}" == "1" ]]; then
  echo "Removing OpenCLAW plugin runtime cache before preflight"
  rm -rf /root/.openclaw/plugin-runtime-deps/openclaw-*
fi

if [[ "${PREFLIGHT}" == "1" ]]; then
  preflight_log="${LOG_DIR}/preflight.log"
  mkdir -p "${PREFLIGHT_WORKSPACE}"
  preflight_model_args=()
  if [[ -n "${AGENT_MODEL}" ]]; then
    preflight_model_args+=(--model "${AGENT_MODEL}")
  fi
  echo "Running OpenCLAW preflight"
  if ! "${OPENCLAW_BIN}" setup --workspace "${PREFLIGHT_WORKSPACE}" >"${preflight_log}" 2>&1; then
    echo "OpenCLAW preflight setup failed. Log: ${preflight_log}" >&2
    exit 1
  fi
  if ! OPENCLAW_PROFILE="memeval-openclaw-preflight" "${OPENCLAW_BIN}" agent \
    --agent "${AGENT}" \
    --local \
    -m "Reply with OK." \
    --session-id "memeval-openclaw-preflight" \
    "${preflight_model_args[@]}" \
    --timeout "${REQUEST_TIMEOUT}" \
    >>"${preflight_log}" 2>&1; then
    echo "OpenCLAW preflight agent failed. Log: ${preflight_log}" >&2
    exit 1
  fi
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

fail_fast_args=()
if [[ "${FAIL_FAST}" == "1" ]]; then
  fail_fast_args+=(--fail-fast)
fi

resume_args=()
if [[ "${RESUME}" == "1" ]]; then
  resume_args+=(--resume)
fi

agent_model_args=()
if [[ -n "${AGENT_MODEL}" ]]; then
  agent_model_args+=(--agent-model "${AGENT_MODEL}")
fi

echo "OpenCLAW session prefix: ${RUN_ID}"

for worker in $(seq 0 $((WORKERS - 1))); do
  worker_start=$((START + worker * CHUNK))
  worker_end=$((worker_start + CHUNK))
  if (( worker_start >= END )); then
    continue
  fi
  if (( worker_end > END )); then
    worker_end="${END}"
  fi

  log_file="${LOG_DIR}/worker_${worker_start}_${worker_end}.log"

  echo "Starting OpenCLAW worker ${worker}: ${worker_start}-${worker_end}"
  "${PYTHON}" "${RUNNER}" \
    --openclaw-bin "${OPENCLAW_BIN}" \
    --agent "${AGENT}" \
    "${agent_model_args[@]}" \
    --session-prefix "${RUN_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --workspace-root "${WORKSPACE_ROOT}" \
    --memory-state-dir "${MEMORY_STATE_DIR}" \
    --phase "${PHASE}" \
    --start "${worker_start}" \
    --end "${worker_end}" \
    --part-size "${PART_SIZE}" \
    --top-k "${TOP_K}" \
    --qa-workers "${QA_WORKERS}" \
    --request-timeout "${REQUEST_TIMEOUT}" \
    --sleep-seconds "${SLEEP_SECONDS}" \
    "${fail_fast_args[@]}" \
    "${resume_args[@]}" \
    >"${log_file}" 2>&1 &
  pids+=("$!")
done

remaining="${#pids[@]}"
while (( remaining > 0 )); do
  set +e
  wait -n
  status="$?"
  set -e
  if (( status != 0 )); then
    echo "An OpenCLAW worker failed with status ${status}. Stopping remaining workers. Logs: ${LOG_DIR}" >&2
    cleanup
    wait || true
    exit "${status}"
  fi
  remaining=$((remaining - 1))
done

echo "All OpenCLAW workers completed. Output: ${OUTPUT_DIR}"
