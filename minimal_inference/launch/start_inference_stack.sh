#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
RADAGENT_ROOT="${ROOT}/radagent"
MAMBA_ROOT_PREFIX="/opt/micromamba"
TOOLS_RUNTIME_PYTHON="${MAMBA_ROOT_PREFIX}/envs/ctchat-tools-runtime/bin/python"
TOOLS_RUNTIME_PATH="${MAMBA_ROOT_PREFIX}/envs/ctchat-tools-runtime/bin:/opt/micromamba/bin:/usr/local/bin:${PATH}"
PYTHONPATH_RUNTIME="${ROOT}:${RADAGENT_ROOT}"
DATA_ROOT="${ROOT}/tmp_3d_radagent/local_data"
SERVICE_START_TIMEOUT_SECONDS="3600"

cd "${ROOT}"

export RADAGENT_DATA_ROOT="${DATA_ROOT}"

mkdir -p \
  "${DATA_ROOT}/uploads" \
  "${DATA_ROOT}/processed/saved_segmentations" \
  "${DATA_ROOT}/processed/saved_classifier" \
  "${DATA_ROOT}/processed/windowed_images" \
  "${DATA_ROOT}/HF_cache" \
  "${RADAGENT_ROOT}/outputs"

PIDS=()

start_bg() {
  echo "[minimal-inference] starting: $*"
  "$@" &
  PIDS+=("$!")
}

start_tools_runtime_python() {
  start_bg env \
    PATH="${TOOLS_RUNTIME_PATH}" \
    PYTHONPATH="${PYTHONPATH_RUNTIME}" \
    RADAGENT_DATA_ROOT="${DATA_ROOT}" \
    "${TOOLS_RUNTIME_PYTHON}" \
    "$@"
}

cleanup() {
  echo "[minimal-inference] stopping child processes"
  local pid

  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local label="$3"
  local timeout_seconds="$4"
  local start_time

  start_time="$(date +%s)"
  while true; do
    if (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
      echo "[minimal-inference] ${label} is reachable on ${host}:${port}"
      return 0
    fi

    if (( "$(date +%s)" - start_time >= timeout_seconds )); then
      echo "[minimal-inference] timed out waiting for ${label} on ${host}:${port}" >&2
      return 1
    fi

    sleep 5
  done
}

trap cleanup EXIT INT TERM

bash "${ROOT}/env/ensure_qwen_vllm_agent_runtime.sh" qwen-vllm-agent-runtime

start_bg bash "${ROOT}/launch/serve_ct_agent.sh"
wait_for_port "127.0.0.1" "8000" "ct-agent server" "${SERVICE_START_TIMEOUT_SECONDS}"

start_bg bash "${ROOT}/launch/start_tool_node_simple.sh"
wait_for_port "127.0.0.1" "7012" "toolbox GPU services" "${SERVICE_START_TIMEOUT_SECONDS}"

start_tools_runtime_python -m app.runtime.mcp_toolbox_proxy --host 0.0.0.0 --port 8080
wait_for_port "127.0.0.1" "8080" "toolbox proxy" "${SERVICE_START_TIMEOUT_SECONDS}"

wait -n || true
