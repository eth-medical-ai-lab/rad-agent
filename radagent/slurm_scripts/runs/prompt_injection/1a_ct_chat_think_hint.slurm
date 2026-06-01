#!/bin/bash -l
#SBATCH --job-name=ct_chat_hallu
#SBATCH --partition=normal
#SBATCH --account=a135
#SBATCH --time=2:59:00
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

nvcc --version
 
#echo $SLURM_NODEID OpenPipe/Qwen3-14B-Instruct

srun bash -c '
    source <CAPSTOR_ROOT>/anaconda3/bin/activate torch2
    PYTHONPATH=<RADAGENT_REPO_ROOT> \
    HF_HOME=$SCRATCH/HF_cache \
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    MASTER_PORT=29500 \
    RANK=${SLURM_PROCID} \
    LOCAL_RANK=${SLURM_LOCALID} \
    WORLD_SIZE=${SLURM_NTASKS} \
    NCCL_TIMEOUT=1800000 \
    python -m evaluation.hallucination_study.run_CT_Chat_inference_hallu --hallu_type='think'
    '
