#!/bin/bash -l
#SBATCH --job-name=gpt-metrics
#SBATCH --partition=debug
#SBATCH --account=a135
#SBATCH --time=0:59:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --uenv=prgenv-gnu/25.6:v2
#SBATCH --view=modules
#SBATCH --output=<SCRATCH_LOG_DIR>/metrics_%j.out
#SBATCH --error=<SCRATCH_LOG_DIR>/metrics_%j.err

module load cuda
module load gcc

export NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
export NODES_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST)

echo $NODES
echo $NODES_IP

srun bash -c "
    source <CAPSTOR_ROOT>/anaconda3/bin/activate RadAgentMel
    PYTHONPATH=<RADAGENT_REPO_ROOT> \
    HF_HOME=$SCRATCH/HF_cache \
    HF_HUB_CACHE=$SCRATCH/HF_cache \
    VLLM_CACHE_ROOT=$SCRATCH/HF_cache \
    LOCAL_NODE_RANK=$SLURM_NODEID \
    SLURM_JOB_ID=$SLURM_JOB_ID \
    BNB_CUDA_VERSION=129 \
    CUDA_LAUNCH_BLOCKING=1 \
    TORCH_USE_CUDA_DSA=1 \
    OMP_NUM_THREADS=1 \
    python evaluation/add_llm_as_judge_scores_to_metrics.py --output-file <RADAGENT_RESULTS_DIR>/baselines/<CT_CHAT_BASELINE_RUN>/<EVAL_SPLIT>/detailed_results.csv
    "
