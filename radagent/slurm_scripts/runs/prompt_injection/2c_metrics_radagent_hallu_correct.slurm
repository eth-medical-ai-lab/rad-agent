#!/bin/bash -l
#SBATCH --job-name=hal_metrics1_ct_chat
#SBATCH --partition=normal
#SBATCH --account=a135
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --output=<SCRATCH_LOG_DIR>/metrics_%j.out
#SBATCH --error=<SCRATCH_LOG_DIR>/metrics_%j.err

nvidia-smi

srun bash -c '
    export WANDB_API_KEY="$(cat ~/.wandb_key)"
    source <CAPSTOR_ROOT>/anaconda3/bin/activate RadAgentMel
    PYTHONPATH=<RADAGENT_REPO_ROOT> \
    HF_HOME=$SCRATCH/HF_cache \
    NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST) \
    MASTER_ADDR=$(echo $NODES | awk "{print \$1}") \
    MASTER_PORT=29500 \
    RANK=${SLURM_PROCID} \
    LOCAL_RANK=${SLURM_LOCALID} \
    WORLD_SIZE=${SLURM_NTASKS} \
    LOCAL_NODE_RANK=$SLURM_NODEID \
    NCCL_TIMEOUT=1800000 \
    python evaluation/process_generated_reports.py --output_folder "<RADAGENT_RESULTS_DIR>/models/<RADAGENT_MODEL_RUN>/checkpoints/<CHECKPOINT_NAME>/<OUTPUT_SUBDIR>" --split="test"
'
