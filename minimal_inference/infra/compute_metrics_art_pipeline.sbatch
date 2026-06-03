#!/bin/bash -l

#SBATCH --job-name=metrics_minimal_art
#SBATCH --partition=normal
#SBATCH --account=a135
#SBATCH --time=0:59:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/iopsstor/scratch/cscs/bgunders/metrics_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/bgunders/metrics_%j.err

set -euo pipefail

ART_ROOT="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/3dragent-agent-mel/radagent"
OUTPUT_FOLDER="/iopsstor/scratch/cscs/bgunders/repos/minimal_inference/radagent/outputs/ctrate_report_generation/minimal_inference/test_0000_0499_array"
SCRATCH_ROOT="${SCRATCH:?missing SCRATCH}"
CACHE_ROOT="${SCRATCH_ROOT}/.cache"
HF_CACHE_ROOT="${CACHE_ROOT}/huggingface"

mkdir -p "${CACHE_ROOT}" "${HF_CACHE_ROOT}" "${HF_CACHE_ROOT}/hub"

nvidia-smi

srun bash -c "
    source /capstor/store/cscs/swissai/a135/wp3-agents/anaconda3/bin/activate RadAgentMel
    PYTHONPATH=${ART_ROOT} \
    PYTHON_DOTENV_DISABLED=1 \
    XDG_CACHE_HOME=${CACHE_ROOT} \
    HF_HOME=${HF_CACHE_ROOT} \
    HF_HUB_CACHE=${HF_CACHE_ROOT}/hub \
    HUGGINGFACE_HUB_CACHE=${HF_CACHE_ROOT}/hub \
    MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
    MASTER_PORT=29500 \
    RANK=\${SLURM_PROCID} \
    LOCAL_RANK=\${SLURM_LOCALID} \
    WORLD_SIZE=\${SLURM_NTASKS} \
    NCCL_TIMEOUT=1800000 \
    python ${ART_ROOT}/evaluation/process_generated_reports.py \
      --output_folder ${OUTPUT_FOLDER} \
      --split test
"
