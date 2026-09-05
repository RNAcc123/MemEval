#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

PYTHON="${PYTHON:-python}"
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
ISOLATE_OPENCLAW_PROFILES="${ISOLATE_OPENCLAW_PROFILES:-1}"
OPENCLAW_PROFILE_BASE="${OPENCLAW_PROFILE_BASE:-memeval-locomo-${RUN_ID}}"
OPENCLAW_CONFIG_SOURCE="${OPENCLAW_CONFIG_SOURCE:-${OPENCLAW_CONFIG_PATH:-${HOME}/.openclaw/openclaw.json}}"
OPENCLAW_PROFILE_PARENT="${OPENCLAW_PROFILE_PARENT:-${HOME}}"
OPENCLAW_RUNTIME_DEPS_SOURCE="${OPENCLAW_RUNTIME_DEPS_SOURCE:-${HOME}/.openclaw/plugin-runtime-deps}"

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

if [[ "${ISOLATE_OPENCLAW_PROFILES}" != "0" && "${ISOLATE_OPENCLAW_PROFILES}" != "1" ]]; then
  echo "ISOLATE_OPENCLAW_PROFILES must be 0 or 1" >&2
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
if [[ "${ISOLATE_OPENCLAW_PROFILES}" == "1" ]]; then
  echo "OpenCLAW profile isolation: ${OPENCLAW_PROFILE_BASE}-worker-N"
fi

safe_profile_name() {
  local raw="$1"
  local safe
  safe="$(printf '%s' "${raw}" | tr -c 'A-Za-z0-9_.-' '_')"
  safe="${safe##_}"
  safe="${safe%%_}"
  printf '%s' "${safe:-memeval-locomo-worker}"
}

init_openclaw_profile() {
  local profile="$1"
  local profile_dir="${OPENCLAW_PROFILE_PARENT}/.openclaw-${profile}"
  if [[ ! -f "${OPENCLAW_CONFIG_SOURCE}" ]]; then
    echo "OpenCLAW config source not found: ${OPENCLAW_CONFIG_SOURCE}" >&2
    exit 1
  fi
  mkdir -p "${profile_dir}"
  cp "${OPENCLAW_CONFIG_SOURCE}" "${profile_dir}/openclaw.json"
  if [[ -d "${OPENCLAW_RUNTIME_DEPS_SOURCE}" ]]; then
    mkdir -p "${profile_dir}/plugin-runtime-deps"
    cp -a "${OPENCLAW_RUNTIME_DEPS_SOURCE}/." "${profile_dir}/plugin-runtime-deps/"
  fi
}

if [[ "${PREFLIGHT}" == "1" ]]; then
  preflight_log="${LOG_DIR}/preflight.log"
  mkdir -p "${PREFLIGHT_WORKSPACE}"
  preflight_model_args=()
  preflight_profile_args=()
  preflight_profile_env="memeval-openclaw-preflight"
  if [[ -n "${AGENT_MODEL}" ]]; then
    preflight_model_args+=(--model "${AGENT_MODEL}")
  fi
  if [[ "${ISOLATE_OPENCLAW_PROFILES}" == "1" ]]; then
    preflight_profile="$(safe_profile_name "${OPENCLAW_PROFILE_BASE}-preflight")"
    init_openclaw_profile "${preflight_profile}"
    preflight_profile_args+=(--profile "${preflight_profile}")
    preflight_profile_env="${preflight_profile}"
  fi
  echo "Running OpenCLAW preflight"
  if ! OPENCLAW_PROFILE="${preflight_profile_env}" "${OPENCLAW_BIN}" "${preflight_profile_args[@]}" setup \
    --workspace "${PREFLIGHT_WORKSPACE}" >"${preflight_log}" 2>&1; then
    echo "OpenCLAW preflight setup failed. Log: ${preflight_log}" >&2
    exit 1
  fi
  if ! OPENCLAW_PROFILE="${preflight_profile_env}" "${OPENCLAW_BIN}" "${preflight_profile_args[@]}" agent \
    --agent "${AGENT}" \
    --local \
    -m "Reply with OK." \
    --session-id "${RUN_ID}-preflight" \
    "${preflight_model_args[@]}" \
    --timeout "${REQUEST_TIMEOUT}" \
    >>"${preflight_log}" 2>&1; then
    echo "OpenCLAW preflight agent failed. Log: ${preflight_log}" >&2
    exit 1
  fi
fi

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
  openclaw_profile_args=()
  if [[ "${ISOLATE_OPENCLAW_PROFILES}" == "1" ]]; then
    worker_profile="$(safe_profile_name "${OPENCLAW_PROFILE_BASE}-worker-${worker}")"
    init_openclaw_profile "${worker_profile}"
    openclaw_profile_args+=(--openclaw-profile "${worker_profile}")
  fi

  echo "Starting OpenCLAW worker ${worker}: ${worker_start}-${worker_end}"
  "${PYTHON}" "${RUNNER}" \
    --openclaw-bin "${OPENCLAW_BIN}" \
    "${openclaw_profile_args[@]}" \
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
