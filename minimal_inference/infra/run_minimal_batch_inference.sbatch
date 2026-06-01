#!/bin/bash
#SBATCH -A a135
#SBATCH --job-name=minimal_ctrate_test_batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --exclude nid007276,nid006192,nid007338,nid007558,nid007576,nid006815,nid007130
#SBATCH --exclusive

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "[sbatch] missing SLURM_SUBMIT_DIR" >&2
  exit 1
fi

SCRATCH_ROOT="${SCRATCH:?missing SCRATCH}"
CACHE_ROOT="${SCRATCH_ROOT}/.cache"
HF_CACHE_ROOT="${CACHE_ROOT}/huggingface"
BUNDLE_ROOT="${SLURM_SUBMIT_DIR}"
LAUNCH="${BUNDLE_ROOT}/launch/run_minimal_batch_inference.sh"
OUTPUT_DIR="${BUNDLE_ROOT}/radagent/outputs/ctrate_report_generation/minimal_inference/test_0000_1303"

mkdir -p "${CACHE_ROOT}" "${HF_CACHE_ROOT}" "${HF_CACHE_ROOT}/hub"

cd "${BUNDLE_ROOT}"

echo "[sbatch] launch script: ${LAUNCH}"
echo "[sbatch] CT-RATE test output dir: ${OUTPUT_DIR}"
echo "[sbatch] XDG cache root: ${CACHE_ROOT}"
echo "[sbatch] Hugging Face cache root: ${HF_CACHE_ROOT}"

srun \
  --environment=tool_box \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --gpus-per-task=4 \
  env \
    XDG_CACHE_HOME="${CACHE_ROOT}" \
    HF_HOME="${HF_CACHE_ROOT}" \
    HF_HUB_CACHE="${HF_CACHE_ROOT}/hub" \
    HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub" \
    BATCH_SPLIT="test" \
    BATCH_START_ID="0" \
    BATCH_END_ID="1303" \
    BATCH_OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "${LAUNCH}"
