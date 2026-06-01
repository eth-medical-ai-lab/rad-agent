#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
RADAGENT_ROOT="${ROOT}/radagent"
MAMBA_ROOT_PREFIX="/opt/micromamba"
TOOLS_RUNTIME_PYTHON="${MAMBA_ROOT_PREFIX}/envs/ctchat-tools-runtime/bin/python"
TOOLS_RUNTIME_PATH="${MAMBA_ROOT_PREFIX}/envs/ctchat-tools-runtime/bin:/opt/micromamba/bin:/usr/local/bin:${PATH}"
PYTHONPATH_RUNTIME="${ROOT}:${RADAGENT_ROOT}"

# Edit these values here if needed.
DATA_ROOT="${ROOT}/tmp_3d_radagent/local_data"
CT_RATE_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset"
CT_CHAT_MODELS_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/trained_models/CT_Chat"
AGENT_MODELS_ROOT="${DATA_ROOT}/HF_cache"
PRECOMPUTED_SEG_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/PROCESSED_FILES_CT_RATE/saved_segmentations"
PRECOMPUTED_CLASSIFIER_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/PROCESSED_FILES_CT_RATE/saved_classifier"
WINDOWED_IMAGES_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/PROCESSED_FILES_CT_RATE/windowed_images"
OUTPUTS_ROOT="${RADAGENT_ROOT}/outputs"
RADCHEST_CT_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/RadChestCT"

BATCH_SPLIT="${BATCH_SPLIT:-val}"
BATCH_START_ID="${BATCH_START_ID:-4000}"
BATCH_END_ID="${BATCH_END_ID:-4999}"
BATCH_MAX_CONCURRENT_CASES="${BATCH_MAX_CONCURRENT_CASES:-4}"
BATCH_MAX_STEPS="${BATCH_MAX_STEPS:-60}"
BATCH_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-${OUTPUTS_ROOT}/ctrate_report_generation/minimal_inference/val_4000_4999}"

MODEL_URL="http://127.0.0.1:8000/v1"
MODEL_NAME="ct-agent"
MODEL_API_KEY="ct-agent"
TOOLBOX_URL="http://127.0.0.1:8080/mcp"
SERVICE_START_TIMEOUT_SECONDS="3600"

cd "${ROOT}"

export RADAGENT_DATA_ROOT="${DATA_ROOT}"
export RADAGENT_CT_RATE_ROOT="${CT_RATE_ROOT}"
export RADAGENT_CT_CHAT_MODELS_ROOT="${CT_CHAT_MODELS_ROOT}"
export RADAGENT_AGENT_MODELS_ROOT="${AGENT_MODELS_ROOT}"
export RADAGENT_PRECOMPUTED_SEG_ROOT="${PRECOMPUTED_SEG_ROOT}"
export RADAGENT_PRECOMPUTED_CLASSIFIER_ROOT="${PRECOMPUTED_CLASSIFIER_ROOT}"
export RADAGENT_WINDOWED_IMAGES_ROOT="${WINDOWED_IMAGES_ROOT}"
export RADAGENT_OUTPUTS_ROOT="${OUTPUTS_ROOT}"
export RADAGENT_RADCHEST_CT_ROOT="${RADCHEST_CT_ROOT}"
export TOOLBOX_URL="${TOOLBOX_URL}"
export CT_AGENT_MODEL_URL="${MODEL_URL}"
export CT_AGENT_MODEL_NAME="${MODEL_NAME}"
export CT_AGENT_API_KEY="${MODEL_API_KEY}"

mkdir -p \
  "${DATA_ROOT}/uploads" \
  "${DATA_ROOT}/processed/saved_segmentations" \
  "${DATA_ROOT}/processed/saved_classifier" \
  "${DATA_ROOT}/processed/windowed_images" \
  "${DATA_ROOT}/HF_cache" \
  "${OUTPUTS_ROOT}" \
  "${BATCH_OUTPUT_DIR}"

if [[ ! -x "${TOOLS_RUNTIME_PYTHON}" ]]; then
  echo "[minimal-batch] missing ctchat-tools-runtime python at ${TOOLS_RUNTIME_PYTHON}" >&2
  exit 1
fi

PIDS=()

start_bg() {
  echo "[minimal-batch] starting: $*"
  "$@" &
  PIDS+=("$!")
}

cleanup() {
  echo "[minimal-batch] stopping child processes"
  local pid

  for pid in "${PIDS[@]:-}"; do
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
      echo "[minimal-batch] ${label} is reachable on ${host}:${port}"
      return 0
    fi

    if (( "$(date +%s)" - start_time >= timeout_seconds )); then
      echo "[minimal-batch] timed out waiting for ${label} on ${host}:${port}" >&2
      return 1
    fi

    sleep 5
  done
}

trap cleanup EXIT INT TERM

start_bg bash "${ROOT}/launch/start_inference_stack.sh"

wait_for_port "127.0.0.1" "8000" "ct-agent server" "${SERVICE_START_TIMEOUT_SECONDS}"
wait_for_port "127.0.0.1" "8080" "toolbox proxy" "${SERVICE_START_TIMEOUT_SECONDS}"

echo "[minimal-batch] running CT-RATE ${BATCH_SPLIT} batch inference"
echo "[minimal-batch] split=${BATCH_SPLIT} start_id=${BATCH_START_ID} end_id=${BATCH_END_ID} max_concurrent=${BATCH_MAX_CONCURRENT_CASES}"
echo "[minimal-batch] output dir: ${BATCH_OUTPUT_DIR}"

env \
  PATH="${TOOLS_RUNTIME_PATH}" \
  PYTHONPATH="${PYTHONPATH_RUNTIME}" \
  RADAGENT_DATA_ROOT="${DATA_ROOT}" \
  RADAGENT_CT_RATE_ROOT="${CT_RATE_ROOT}" \
  RADAGENT_CT_CHAT_MODELS_ROOT="${CT_CHAT_MODELS_ROOT}" \
  RADAGENT_AGENT_MODELS_ROOT="${AGENT_MODELS_ROOT}" \
  RADAGENT_PRECOMPUTED_SEG_ROOT="${PRECOMPUTED_SEG_ROOT}" \
  RADAGENT_PRECOMPUTED_CLASSIFIER_ROOT="${PRECOMPUTED_CLASSIFIER_ROOT}" \
  RADAGENT_WINDOWED_IMAGES_ROOT="${WINDOWED_IMAGES_ROOT}" \
  RADAGENT_OUTPUTS_ROOT="${OUTPUTS_ROOT}" \
  RADAGENT_RADCHEST_CT_ROOT="${RADCHEST_CT_ROOT}" \
  "${TOOLS_RUNTIME_PYTHON}" \
  -m app.cli.minimal_ct_agent_batch \
  --split "${BATCH_SPLIT}" \
  --start-id "${BATCH_START_ID}" \
  --end-id "${BATCH_END_ID}" \
  --max-concurrent-cases "${BATCH_MAX_CONCURRENT_CASES}" \
  --output-dir "${BATCH_OUTPUT_DIR}" \
  --model-url "${MODEL_URL}" \
  --model-name "${MODEL_NAME}" \
  --api-key "${MODEL_API_KEY}" \
  --toolbox-url "${TOOLBOX_URL}" \
  --max-steps "${BATCH_MAX_STEPS}"
