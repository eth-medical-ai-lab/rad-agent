#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
RADAGENT_ROOT="${ROOT}/radagent"
MAMBA_ROOT_PREFIX="/opt/micromamba"
AGENT_RUNTIME_PREFIX="${MAMBA_ROOT_PREFIX}/envs/qwen-vllm-agent-runtime"
VLLM_BIN="${AGENT_RUNTIME_PREFIX}/bin/vllm"
ENV_PATH="${AGENT_RUNTIME_PREFIX}/bin:/opt/micromamba/bin:/usr/local/bin:${PATH}"
PYTHONPATH_RUNTIME="${ROOT}:${RADAGENT_ROOT}"
DATA_ROOT="${ROOT}/tmp_3d_radagent/local_data"

cd "${ROOT}"
mkdir -p "${DATA_ROOT}"

bash "${ROOT}/env/ensure_qwen_vllm_agent_runtime.sh" qwen-vllm-agent-runtime

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "[ct-agent-server] missing vllm executable at ${VLLM_BIN}" >&2
  exit 1
fi

# Edit these for your node / model.
CT_AGENT_BASE_MODEL="OpenPipe/Qwen3-14B-Instruct"
CT_AGENT_LORA_ADAPTER="RadAgent/radagent-qwen3-14b-lora"
CT_AGENT_SERVED_NAME="ct-agent-base"
CT_AGENT_LORA_NAME="ct-agent"
CT_AGENT_PORT="8000"
CT_AGENT_DEVICE="0"
CT_AGENT_TENSOR_PARALLEL_SIZE="1"
CT_AGENT_MAX_MODEL_LEN="32768"
CT_AGENT_GPU_MEMORY_UTILIZATION="0.8"
CT_AGENT_DTYPE="bfloat16"
CT_AGENT_MAX_LORA_RANK="16"
CT_AGENT_API_KEY="ct-agent"

echo "[ct-agent-server] serving base model ${CT_AGENT_BASE_MODEL} with LoRA ${CT_AGENT_LORA_ADAPTER} as '${CT_AGENT_LORA_NAME}' on GPU(s) '${CT_AGENT_DEVICE}' with max model len ${CT_AGENT_MAX_MODEL_LEN}"

exec env \
  PATH="${ENV_PATH}" \
  CUDA_VISIBLE_DEVICES="${CT_AGENT_DEVICE}" \
  CUDA_LAUNCH_BLOCKING=1 \
  TOKENIZERS_PARALLELISM=false \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${PYTHONPATH_RUNTIME}" \
  RADAGENT_DATA_ROOT="${DATA_ROOT}" \
  "${VLLM_BIN}" serve "${CT_AGENT_BASE_MODEL}" \
  --host 0.0.0.0 \
  --port "${CT_AGENT_PORT}" \
  --served-model-name "${CT_AGENT_SERVED_NAME}" \
  --enable-lora \
  --lora-modules "${CT_AGENT_LORA_NAME}=${CT_AGENT_LORA_ADAPTER}" \
  --max-lora-rank "${CT_AGENT_MAX_LORA_RANK}" \
  --tensor-parallel-size "${CT_AGENT_TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${CT_AGENT_MAX_MODEL_LEN}" \
  --dtype "${CT_AGENT_DTYPE}" \
  --gpu-memory-utilization "${CT_AGENT_GPU_MEMORY_UTILIZATION}" \
  --api-key "${CT_AGENT_API_KEY}" \
  --trust-remote-code
