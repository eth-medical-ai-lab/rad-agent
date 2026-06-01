#!/bin/bash -l
#SBATCH --job-name=<SLURM_JOB_NAME>
#SBATCH --partition=normal
#SBATCH --account=a135
#SBATCH --time=11:59:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=<SCRATCH_LOG_DIR>/v5_report_%j.out
#SBATCH --error=<SCRATCH_LOG_DIR>/v5_report_%j.err
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=modules
#SBATCH --exclude=nid006845

module load cuda
module load gcc

nvidia-smi

nvcc --version

srun bash -c '
    NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    source <CAPSTOR_ROOT>/anaconda3/bin/activate RadAgentMelRL
    PYTHONPATH=<RADAGENT_REPO_ROOT> \
    HF_HOME=$SCRATCH/HF_cache \
    VLLM_CACHE_ROOT=$SCRATCH/HF_cache \
    LOCAL_NODE_RANK=$SLURM_NODEID \
    NODES_IP=$NODES \
    BNB_CUDA_VERSION=129 \
    CUDA_LAUNCH_BLOCKING=1 \
    TORCH_USE_CUDA_DSA=1 \
    OMP_NUM_THREADS=1 \
    CUDA_HOME='<CUDA_HOME>' \
    SLURM_JOB_ID=$SLURM_JOB_ID \
    python agents/main_ct_rate.py --model_name='OpenPipe/Qwen3-14B-Instruct' --mode='train' --task='report_generation' --subdir='<TRAINING_RUN_NAME>' --batch_size=6 --n_rollout_per_group=8 --agent_type='v8c' --lambda_manual_tool_judge=0.2 --lambda_tool_success=0.1 --lambda_llm_judge_f1=1.0 --lambda_llm_judge_prec_recall=0.0 --lambda_f1_text_classifier=1.0 --lambda_green=0.0 --start_id=100 --beta=0.001 --shuffle_dataset --lambda_trajectory_judge=0.2
'

# continued from 89 with v8b
