#!/bin/bash

#SBATCH -A a135
#SBATCH --job-name=minimal_ctrate_test_batch_array
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --exclude nid007276,nid006192,nid007338,nid007558,nid007576,nid006815,nid007130
#SBATCH --exclusive

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "[sbatch-array] missing SLURM_SUBMIT_DIR" >&2
  exit 1
fi

SCRATCH_ROOT="${SCRATCH:?missing SCRATCH}"
CACHE_ROOT="${SCRATCH_ROOT}/.cache"
HF_CACHE_ROOT="${CACHE_ROOT}/huggingface"
BUNDLE_ROOT="${SLURM_SUBMIT_DIR}"
LAUNCH="${BUNDLE_ROOT}/launch/run_minimal_batch_inference.sh"
ARRAY_BASE_START="0"
ARRAY_BASE_END="499"
CHUNK_SIZE="100"
ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:?missing SLURM_ARRAY_TASK_COUNT}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
TASK_START_ID="$((ARRAY_BASE_START + TASK_INDEX * CHUNK_SIZE))"
TASK_END_ID="$((TASK_START_ID + CHUNK_SIZE - 1))"
OUTPUT_DIR="${BUNDLE_ROOT}/radagent/outputs/ctrate_report_generation/minimal_inference/test_0000_0499_array"

if (( TASK_INDEX < 0 || TASK_INDEX >= ARRAY_TASK_COUNT )); then
  echo "[sbatch-array] invalid array task index ${TASK_INDEX}" >&2
  exit 1
fi

if (( TASK_START_ID > ARRAY_BASE_END )); then
  echo "[sbatch-array] task ${TASK_INDEX} starts beyond ARRAY_BASE_END=${ARRAY_BASE_END}" >&2
  exit 1
fi

if (( TASK_END_ID > ARRAY_BASE_END )); then
  TASK_END_ID="${ARRAY_BASE_END}"
fi

mkdir -p "${CACHE_ROOT}" "${HF_CACHE_ROOT}" "${HF_CACHE_ROOT}/hub"

cd "${BUNDLE_ROOT}"

echo "[sbatch-array] launch script: ${LAUNCH}"
echo "[sbatch-array] task $((TASK_INDEX + 1))/${ARRAY_TASK_COUNT} -> indices ${TASK_START_ID}-${TASK_END_ID}"
echo "[sbatch-array] output dir: ${OUTPUT_DIR}"
echo "[sbatch-array] XDG cache root: ${CACHE_ROOT}"
echo "[sbatch-array] Hugging Face cache root: ${HF_CACHE_ROOT}"

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
    BATCH_START_ID="${TASK_START_ID}" \
    BATCH_END_ID="${TASK_END_ID}" \
    BATCH_OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "${LAUNCH}"
