#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
RADAGENT_ROOT="${ROOT}/radagent"
MAMBA_ROOT_PREFIX="/opt/micromamba"
PYTHONPATH_RUNTIME="${ROOT}:${RADAGENT_ROOT}"

# Edit these paths for your node.

DATA_ROOT="${ROOT}/tmp_3d_radagent/local_data"
CT_RATE_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset"
CT_CHAT_MODELS_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/trained_models/CT_Chat"
AGENT_MODELS_ROOT="${DATA_ROOT}/HF_cache"
PRECOMPUTED_SEG_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/PROCESSED_FILES_CT_RATE/saved_segmentations"
PRECOMPUTED_CLASSIFIER_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/PROCESSED_FILES_CT_RATE/saved_classifier"
WINDOWED_IMAGES_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/PROCESSED_FILES_CT_RATE/windowed_images"
OUTPUTS_ROOT="${RADAGENT_ROOT}/outputs"
RADCHEST_CT_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/RadChestCT"

export RADAGENT_DATA_ROOT="${DATA_ROOT}"
export RADAGENT_CT_RATE_ROOT="${CT_RATE_ROOT}"
export RADAGENT_CT_CHAT_MODELS_ROOT="${CT_CHAT_MODELS_ROOT}"
export RADAGENT_AGENT_MODELS_ROOT="${AGENT_MODELS_ROOT}"
export RADAGENT_PRECOMPUTED_SEG_ROOT="${PRECOMPUTED_SEG_ROOT}"
export RADAGENT_PRECOMPUTED_CLASSIFIER_ROOT="${PRECOMPUTED_CLASSIFIER_ROOT}"
export RADAGENT_WINDOWED_IMAGES_ROOT="${WINDOWED_IMAGES_ROOT}"
export RADAGENT_OUTPUTS_ROOT="${OUTPUTS_ROOT}"
export RADAGENT_RADCHEST_CT_ROOT="${RADCHEST_CT_ROOT}"

cd "${ROOT}"
mkdir -p "${DATA_ROOT}" "${AGENT_MODELS_ROOT}" "${OUTPUTS_ROOT}"

bash "${ROOT}/env/ensure_qwen_vllm_agent_runtime.sh" qwen-vllm-agent-runtime

cd "${RADAGENT_ROOT}"

echo "[toolbox-node] using data root '${DATA_ROOT}'"
echo "[toolbox-node] using CT-RATE root '${CT_RATE_ROOT}'"
echo "[toolbox-node] using CT-Chat models root '${CT_CHAT_MODELS_ROOT}'"

# Edit these assignments for your node.
# Use:
#   cpu   -> force CPU
#   0     -> one GPU
#   0,1   -> multi-GPU tool
WINDOWING_TOOL_DEVICE="cpu"
BIGGEST_SLICE_TOOL_DEVICE="cpu"
SLICE_TOOL_DEVICE="cpu"
SEVERAL_SLICES_TOOL_DEVICE="cpu"
ANATOMY_SEGMENTATION_TOOL_DEVICE="3"
EFFUSION_SEGMENTATION_TOOL_DEVICE="3"
DISEASE_CLASSIFIER_TOOL_DEVICE="3"
REPORT_GENERATION_TOOL_DEVICE="3"
SLICE_VQA_TOOL_DEVICE="1"
CT_VQA_TOOL_DEVICE="2"
TOTALSEG_DEVICE="gpu"
TOTALSEG_USE_FAST="0"
WINDOWING_TOOL_PORT="7013"
BIGGEST_SLICE_TOOL_PORT="7014"
SLICE_TOOL_PORT="7017"
SEVERAL_SLICES_TOOL_PORT="7018"
ANATOMY_SEGMENTATION_TOOL_PORT="7011"
EFFUSION_SEGMENTATION_TOOL_PORT="7012"
DISEASE_CLASSIFIER_TOOL_PORT="7010"
REPORT_GENERATION_TOOL_PORT="7006"
SLICE_VQA_TOOL_PORT="7016"
CT_VQA_TOOL_PORT="6009"
TOOL_SERVER_REGISTRY_PATH="${DATA_ROOT}/tool_server_registry.json"
TOOL_START_TIMEOUT_SECONDS="3600"

PIDS=()

write_runtime_registry() {
  cat > "${TOOL_SERVER_REGISTRY_PATH}" <<EOF
{
  "windowing_tool": {
    "addr": "http://127.0.0.1:${WINDOWING_TOOL_PORT}/mcp",
    "port": "${WINDOWING_TOOL_PORT}",
    "env": "ctchat-tools-runtime",
    "device": "${WINDOWING_TOOL_DEVICE}"
  },
  "biggest_slice_selection_tool": {
    "addr": "http://127.0.0.1:${BIGGEST_SLICE_TOOL_PORT}/mcp",
    "port": "${BIGGEST_SLICE_TOOL_PORT}",
    "env": "ctchat-tools-runtime",
    "device": "${BIGGEST_SLICE_TOOL_DEVICE}"
  },
  "extract_slices_from_ct": {
    "addr": "http://127.0.0.1:${SLICE_TOOL_PORT}/mcp",
    "port": "${SLICE_TOOL_PORT}",
    "env": "ctchat-tools-runtime",
    "device": "${SLICE_TOOL_DEVICE}"
  },
  "get_several_slices_from_segmentation": {
    "addr": "http://127.0.0.1:${SEVERAL_SLICES_TOOL_PORT}/mcp",
    "port": "${SEVERAL_SLICES_TOOL_PORT}",
    "env": "ctchat-tools-runtime",
    "device": "${SEVERAL_SLICES_TOOL_DEVICE}"
  },
  "anatomy_segmentation_tool": {
    "addr": "http://127.0.0.1:${ANATOMY_SEGMENTATION_TOOL_PORT}/mcp",
    "port": "${ANATOMY_SEGMENTATION_TOOL_PORT}",
    "env": "ctchat-tools-runtime",
    "device": "${ANATOMY_SEGMENTATION_TOOL_DEVICE}"
  },
  "effusion_segmentation_tool": {
    "addr": "http://127.0.0.1:${EFFUSION_SEGMENTATION_TOOL_PORT}/mcp",
    "port": "${EFFUSION_SEGMENTATION_TOOL_PORT}",
    "env": "ctchat-tools-runtime",
    "device": "${EFFUSION_SEGMENTATION_TOOL_DEVICE}"
  },
  "disease_classifier_tool": {
    "addr": "http://127.0.0.1:${DISEASE_CLASSIFIER_TOOL_PORT}/mcp",
    "port": "${DISEASE_CLASSIFIER_TOOL_PORT}",
    "env": "ctclip-runtime",
    "device": "${DISEASE_CLASSIFIER_TOOL_DEVICE}"
  },
  "report_generation_tool": {
    "addr": "http://127.0.0.1:${REPORT_GENERATION_TOOL_PORT}/mcp",
    "port": "${REPORT_GENERATION_TOOL_PORT}",
    "env": "ctchat-tools-runtime",
    "device": "${REPORT_GENERATION_TOOL_DEVICE}"
  },
  "slice_vqa_tool": {
    "addr": "http://127.0.0.1:${SLICE_VQA_TOOL_PORT}/mcp",
    "port": "${SLICE_VQA_TOOL_PORT}",
    "env": "qwen-vllm-agent-runtime",
    "device": "${SLICE_VQA_TOOL_DEVICE}"
  },
  "ct_vqa_tool": {
    "addr": "http://127.0.0.1:${CT_VQA_TOOL_PORT}/mcp",
    "port": "${CT_VQA_TOOL_PORT}",
    "env": "ctchat-tools-runtime",
    "device": "${CT_VQA_TOOL_DEVICE}"
  }
}
EOF
  echo "[toolbox-node] wrote runtime registry to '${TOOL_SERVER_REGISTRY_PATH}'"
}

env_python() {
  local env_name="$1"
  echo "${MAMBA_ROOT_PREFIX}/envs/${env_name}/bin/python"
}

start_tool() {
  local name="$1"
  local env_name="$2"
  local script_path="$3"
  local port="$4"
  local device="$5"
  shift 5
  local python_bin
  local env_path
  local cuda_visible_devices
  local -a script_args

  script_args=("$@")

  python_bin="$(env_python "${env_name}")"
  env_path="$(dirname "${python_bin}"):/opt/micromamba/bin:/usr/local/bin:${PATH}"

  if [[ ! -x "${python_bin}" ]]; then
    echo "[toolbox-node] missing Python env for ${name} at ${python_bin}" >&2
    exit 1
  fi

  echo "[toolbox-node] starting ${name} on device '${device}' with env '${env_name}'"

  if [[ "${device}" == "cpu" || "${device}" == "none" ]]; then
    cuda_visible_devices=""
  else
    cuda_visible_devices="${device}"
  fi

  env \
    PATH="${env_path}" \
    CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PYTHONPATH_RUNTIME}" \
    "${python_bin}" "${script_path}" --host 0.0.0.0 --port "${port}" "${script_args[@]}" &

  PIDS+=("$!")
}

cleanup() {
  echo "[toolbox-node] stopping child processes"
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
      echo "[toolbox-node] ${label} is reachable on ${host}:${port}"
      return 0
    fi

    if (( "$(date +%s)" - start_time >= timeout_seconds )); then
      echo "[toolbox-node] timed out waiting for ${label} on ${host}:${port}" >&2
      return 1
    fi

    sleep 5
  done
}

trap cleanup EXIT INT TERM

write_runtime_registry

VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=TORCH_SDPA \
  start_tool \
  "slice_vqa_tool" \
  "qwen-vllm-agent-runtime" \
  "tools/slice_vqa_tool.py" \
  "${SLICE_VQA_TOOL_PORT}" \
  "${SLICE_VQA_TOOL_DEVICE}"
wait_for_port "127.0.0.1" "${SLICE_VQA_TOOL_PORT}" "slice_vqa_tool" "${TOOL_START_TIMEOUT_SECONDS}"
start_tool "ct_vqa_tool" "ctchat-tools-runtime" "tools/ct_chat.py" "${CT_VQA_TOOL_PORT}" "${CT_VQA_TOOL_DEVICE}"
wait_for_port "127.0.0.1" "${CT_VQA_TOOL_PORT}" "ct_vqa_tool" "${TOOL_START_TIMEOUT_SECONDS}"

start_tool "windowing_tool" "ctchat-tools-runtime" "tools/windowing_tool.py" "${WINDOWING_TOOL_PORT}" "${WINDOWING_TOOL_DEVICE}"
start_tool "biggest_slice_selection_tool" "ctchat-tools-runtime" "tools/biggest_slice_selection_tool.py" "${BIGGEST_SLICE_TOOL_PORT}" "${BIGGEST_SLICE_TOOL_DEVICE}"
start_tool "extract_slices_from_ct" "ctchat-tools-runtime" "tools/select_ct_slices.py" "${SLICE_TOOL_PORT}" "${SLICE_TOOL_DEVICE}"
start_tool "get_several_slices_from_segmentation" "ctchat-tools-runtime" "tools/equidistant_slice_selection_seg.py" "${SEVERAL_SLICES_TOOL_PORT}" "${SEVERAL_SLICES_TOOL_DEVICE}"

start_tool "report_generation_tool" "ctchat-tools-runtime" "tools/report_generator.py" "${REPORT_GENERATION_TOOL_PORT}" "${REPORT_GENERATION_TOOL_DEVICE}"
wait_for_port "127.0.0.1" "${REPORT_GENERATION_TOOL_PORT}" "report_generation_tool" "${TOOL_START_TIMEOUT_SECONDS}"

start_tool "disease_classifier_tool" "ctclip-runtime" "tools/disease_classifier.py" "${DISEASE_CLASSIFIER_TOOL_PORT}" "${DISEASE_CLASSIFIER_TOOL_DEVICE}"
wait_for_port "127.0.0.1" "${DISEASE_CLASSIFIER_TOOL_PORT}" "disease_classifier_tool" "${TOOL_START_TIMEOUT_SECONDS}"

export TOTALSEG_DEVICE TOTALSEG_USE_FAST
start_tool "anatomy_segmentation_tool" "ctchat-tools-runtime" "tools/anatomy_segmentation_tool.py" "${ANATOMY_SEGMENTATION_TOOL_PORT}" "${ANATOMY_SEGMENTATION_TOOL_DEVICE}"
wait_for_port "127.0.0.1" "${ANATOMY_SEGMENTATION_TOOL_PORT}" "anatomy_segmentation_tool" "${TOOL_START_TIMEOUT_SECONDS}"

export TOTALSEG_DEVICE TOTALSEG_USE_FAST
start_tool "effusion_segmentation_tool" "ctchat-tools-runtime" "tools/effusion_segmentation_tool.py" "${EFFUSION_SEGMENTATION_TOOL_PORT}" "${EFFUSION_SEGMENTATION_TOOL_DEVICE}"
wait_for_port "127.0.0.1" "${EFFUSION_SEGMENTATION_TOOL_PORT}" "effusion_segmentation_tool" "${TOOL_START_TIMEOUT_SECONDS}"

wait -n
