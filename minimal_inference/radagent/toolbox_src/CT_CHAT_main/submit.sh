#!/bin/bash -l
#SBATCH --job-name=grpo33
#SBATCH --partition=normal
#SBATCH --account=a135
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=/iopsstor/scratch/cscs/mroschewitz/test_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mroschewitz/test_%j.err
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=modules
#SBATCH --exclude=nid007666

module load cuda
module load gcc
nvidia-smi

srun bash -c '
    source /capstor/store/cscs/swissai/a135/wp3-agents/anaconda3/bin/activate torch2rl
    NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    PYTHONPATH=/capstor/store/cscs/swissai/a135/wp3-agents/workspace/3dragent-agent-mel/radagent \
    HF_HOME=$SCRATCH/HF_cache \
    VLLM_CACHE_ROOT=$SCRATCH/HF_cache \
    LOCAL_NODE_RANK=$SLURM_NODEID \
    NODES_IP=$NODES \
    BNB_CUDA_VERSION=129 \
    SLURM_JOB_ID=$SLURM_JOB_ID \
    python train_grpo.py
    '
