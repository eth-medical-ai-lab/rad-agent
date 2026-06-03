#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-qwen-vllm-agent-runtime}"
MAMBA_ROOT_PREFIX="/opt/micromamba"
ENV_PREFIX="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
PIP_BIN="${ENV_PREFIX}/bin/pip"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0}"
export TORCH_VERSION="${TORCH_VERSION:-2.8.0+cu129}"
ENV_PATH="${ENV_PREFIX}/bin:${CUDA_HOME}/bin:/usr/local/bin:${PATH}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export MAX_JOBS="${MAX_JOBS:-16}"

report_cuda_toolchain() {
  if ! command -v nvcc >/dev/null 2>&1; then
    cat >&2 <<'EOF'
[qwen-vllm-agent-bootstrap] nvcc was not found while building xFormers.
Make sure the build environment exposes a CUDA toolkit with nvcc, and export
CUDA_HOME to that toolkit root before running this bootstrap script.
EOF
    return 0
  fi

  if [[ ! -d "${CUDA_HOME}" ]]; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
  fi
}

validate_cuda_python_env() {
  env PATH="${ENV_PATH}" CUDA_HOME="${CUDA_HOME}" "${PYTHON_BIN}" - <<'PY'
import shutil
import torch
from torch.utils.cpp_extension import CUDA_HOME as ext_cuda_home

print(f"[qwen-vllm-agent-bootstrap] torch.version.cuda={torch.version.cuda}")
print(f"[qwen-vllm-agent-bootstrap] torch cpp_extension CUDA_HOME={ext_cuda_home}")
print(f"[qwen-vllm-agent-bootstrap] nvcc={shutil.which('nvcc')}")

issues = []
if torch.version.cuda is None:
    issues.append("PyTorch is not a CUDA build.")
if shutil.which("nvcc") is None:
    issues.append("nvcc is not visible in PATH.")
if ext_cuda_home is None:
    issues.append("torch.utils.cpp_extension could not resolve CUDA_HOME.")

for issue in issues:
    print(f"[qwen-vllm-agent-bootstrap] WARNING: {issue}")
PY
}

validate_vllm_cuda_build() {
  env PATH="${ENV_PATH}" CUDA_HOME="${CUDA_HOME}" "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import importlib.util
import torch
import vllm

module_version = getattr(vllm, "__version__", "")
dist_version = md.version("vllm")
custom_ops_spec = importlib.util.find_spec("vllm._C")

print(f"[qwen-vllm-agent-bootstrap] vllm.__version__={module_version}")
print(f"[qwen-vllm-agent-bootstrap] vllm dist version={dist_version}")
print(f"[qwen-vllm-agent-bootstrap] vllm._C spec={custom_ops_spec}")

issues = []
if torch.version.cuda is None:
    issues.append("PyTorch is not a CUDA build.")
if custom_ops_spec is None:
    issues.append("vLLM compiled extension vllm._C is missing.")
else:
    try:
        import vllm._C  # noqa: F401
        print("[qwen-vllm-agent-bootstrap] imported vllm._C successfully")
    except Exception as exc:
        issues.append(f"Failed to import vllm._C: {exc}")

for issue in issues:
    print(f"[qwen-vllm-agent-bootstrap] WARNING: {issue}")
PY
}

validate_xformers_cuda() {
  local info

  info="$(env PATH="${ENV_PATH}" XFORMERS_MORE_DETAILS=1 "${PYTHON_BIN}" -m xformers.info 2>&1 || true)"
  printf '%s\n' "${info}"

  if printf '%s\n' "${info}" | grep -Fq "xFormers wasn't build with CUDA support" \
    || printf '%s\n' "${info}" | grep -Eq '^build\.cuda_version:[[:space:]]+None$'; then
    cat >&2 <<EOF
[qwen-vllm-agent-bootstrap] xFormers installed without CUDA kernels.
This usually means the bootstrap ran without a CUDA toolkit in PATH, or with an
unset/misconfigured CUDA_HOME (${CUDA_HOME:-unset}).

If this build runs on a host where GPUs are not visible during image build,
xFormers may also need TORCH_CUDA_ARCH_LIST to be set explicitly so it knows
which CUDA architectures to compile for.
EOF
    return 1
  fi
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[qwen-vllm-agent-bootstrap] missing Python env at ${ENV_PREFIX}" >&2
  exit 1
fi

report_cuda_toolchain
echo "[qwen-vllm-agent-bootstrap] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "[qwen-vllm-agent-bootstrap] TORCH_VERSION=${TORCH_VERSION}"

env PATH="${ENV_PATH}" "${PIP_BIN}" install \
  torch=="${TORCH_VERSION}" torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129
validate_cuda_python_env
env PATH="${ENV_PATH}" "${PIP_BIN}" install openpipe-art==0.5.3
env PATH="${ENV_PATH}" CUDA_HOME="${CUDA_HOME}" MAX_JOBS="${MAX_JOBS}" "${PIP_BIN}" install -v \
  vllm==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu129
validate_vllm_cuda_build
env PATH="${ENV_PATH}" CUDA_HOME="${CUDA_HOME}" MAX_JOBS="${MAX_JOBS}" "${PIP_BIN}" install -v --no-build-isolation 'git+https://github.com/facebookresearch/xformers@v0.0.30'
validate_xformers_cuda
# Make sure the install survives after the temporary checkout is cleaned up.
env PATH="${ENV_PATH}" "${PYTHON_BIN}" -c "import vllm"

env PATH="${ENV_PATH}" "${PIP_BIN}" install polars tblib
env PATH="${ENV_PATH}" "${PIP_BIN}" install \
  "git+https://github.com/pytorch/torchtune.git@2344509cf83bd886538fe3e8263e5145d1afb5c2"
env PATH="${ENV_PATH}" "${PIP_BIN}" install \
  torchao==0.13.0 peft hf-xet bitsandbytes \
  unsloth==2025.10.3 unsloth-zoo==2025.10.3 trl==0.20.0
env PATH="${ENV_PATH}" "${PIP_BIN}" install \
  accelerate==1.7.0 awscli setproctitle wandb==0.21.0 transformers==4.53.2
env PATH="${ENV_PATH}" "${PIP_BIN}" install \
  nbclient pytest nbmake gql==3.5.3
