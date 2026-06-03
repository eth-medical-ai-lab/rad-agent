#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="slurm_logs/minimal_ctrate_test_0000_0499_array_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_ROOT}"

echo "[submit-array] sbatch script: infra/run_minimal_batch_inference_array.sbatch"
echo "[submit-array] logs: ${LOG_ROOT}"

sbatch \
  --output="${LOG_ROOT}/%A_%a.log" \
  infra/run_minimal_batch_inference_array.sbatch
