#!/bin/bash -l
#SBATCH --job-name=add_faithful_label
#SBATCH --partition=normal
#SBATCH --account=a135
#SBATCH --time=0:45:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=<SCRATCH_LOG_DIR>/report_%j.out
#SBATCH --error=<SCRATCH_LOG_DIR>/report_%j.err
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=modules

module load cuda
module load gcc

nvidia-smi
which gcc
nvcc --version
 
#echo $SLURM_NODEID OpenPipe/Qwen3-14B-Instruct

srun bash -c '
    export WANDB_API_KEY="$(cat ~/.wandb_key)"
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
    python evaluation/hallucination_study/get_hint_acknowledgement_labels.py \
        --hint_is_correct true \
        --system_type ct-chat \
        --prompt_type think \
        --judge-type strict
'
