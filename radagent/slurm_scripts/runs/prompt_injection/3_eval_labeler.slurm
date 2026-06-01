#!/bin/bash -l
#SBATCH --job-name=eval_labeler
#SBATCH --partition=normal
#SBATCH --account=a135
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=<SCRATCH_LOG_DIR>/eval_labeler_%j.out
#SBATCH --error=<SCRATCH_LOG_DIR>/eval_labeler_%j.err
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=modules

module load cuda
module load gcc

nvidia-smi
which gcc
nvcc --version

srun bash -c '
    export WANDB_API_KEY="$(cat ~/.wandb_key)"
    export OPENAI_API_KEY="$(cat ~/.openai_api_key)"
    source <CAPSTOR_ROOT>/anaconda3/bin/activate faithful
    NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    PYTHONPATH=<RADAGENT_REPO_ROOT> \
    HF_HOME=$SCRATCH/HF_cache \
    VLLM_CACHE_ROOT=$SCRATCH/HF_cache \
    LOCAL_NODE_RANK=$SLURM_NODEID \
    NODES_IP=$NODES \
    BNB_CUDA_VERSION=129 \
    CUDA_LAUNCH_BLOCKING=1 \
    TORCH_USE_CUDA_DSA=1 \
    OMP_NUM_THREADS=4 \
    CUDA_HOME="<CUDA_HOME>" \
    OPENAI_MODEL="gpt-5.4-mini-2026-03-17" \
    python -m evaluation.hallucination_study.evaluate_labeler \
    --system-types CT-Chat RadAgent --subset-size-per-label 100 --openai-model gpt-5.4-mini-2026-03-17
'
