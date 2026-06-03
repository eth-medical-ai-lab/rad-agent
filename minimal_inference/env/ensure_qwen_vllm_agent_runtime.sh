#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
ENV_NAME="${1:-qwen-vllm-agent-runtime}"
MAMBA_ROOT_PREFIX="/opt/micromamba"
ENV_PREFIX="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
BOOTSTRAP_SCRIPT="${ROOT}/env/bootstrap_qwen_vllm_agent_runtime.sh"
LOCK_ROOT="${ROOT}/tmp_3d_radagent/local_data/.locks"
LOCK_DIR="${LOCK_ROOT}/qwen-vllm-agent-bootstrap"
WAIT_SECONDS="${QWEN_VLLM_AGENT_BOOTSTRAP_WAIT_SECONDS:-1800}"

has_vllm() {
  "${PYTHON_BIN}" -c "import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec('vllm') else 1)" \
    >/dev/null 2>&1
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[qwen-vllm-agent-runtime] missing Python env at ${ENV_PREFIX}" >&2
  exit 1
fi

if has_vllm; then
  exit 0
fi

mkdir -p "${LOCK_ROOT}"

if mkdir "${LOCK_DIR}" 2>/dev/null; then
  trap 'rmdir "${LOCK_DIR}" >/dev/null 2>&1 || true' EXIT
  if has_vllm; then
    exit 0
  fi
  echo "[qwen-vllm-agent-runtime] vllm missing in ${ENV_NAME}; running bootstrap"
  bash "${BOOTSTRAP_SCRIPT}" "${ENV_NAME}"
  if ! has_vllm; then
    echo "[qwen-vllm-agent-runtime] bootstrap finished, but vllm is still unavailable" >&2
    exit 1
  fi
  exit 0
fi

echo "[qwen-vllm-agent-runtime] waiting for another bootstrap process to finish"

elapsed=0
while (( elapsed < WAIT_SECONDS )); do
  if has_vllm; then
    exit 0
  fi
  if [[ ! -d "${LOCK_DIR}" ]]; then
    break
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

if has_vllm; then
  exit 0
fi

echo "[qwen-vllm-agent-runtime] vllm is still missing after waiting for bootstrap" >&2
exit 1
