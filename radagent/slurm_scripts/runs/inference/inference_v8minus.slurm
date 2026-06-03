#!/bin/bash -l
#SBATCH --job-name=v8c_base_inference
#SBATCH --partition=normal
#SBATCH --account=a135
#SBATCH --time=11:59:00
#SBATCH --nodes=2
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
    export WANDB_API_KEY="$(cat ~/.wandb_key)"
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
    OMP_NUM_THREADS=4 \
    CUDA_HOME='<CUDA_HOME>' \
    python agents/main_ct_rate.py --model_name='OpenPipe/Qwen3-14B-Instruct' --mode='val' --task='report_generation' --batch_size=128 --start_id=0 --end_id=1000 --inference_model_name='<RADAGENT_RESULTS_DIR>/models/<RADAGENT_MODEL_RUN>/checkpoints/<CHECKPOINT_NAME>' --agent_type='v8minus'
    '
