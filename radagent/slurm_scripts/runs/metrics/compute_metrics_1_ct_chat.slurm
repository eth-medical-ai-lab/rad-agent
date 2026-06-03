#!/bin/bash -l
#SBATCH --job-name=metrics1_ct_chat
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

srun bash -c "
    source <CAPSTOR_ROOT>/anaconda3/bin/activate RadAgentMel
    PYTHONPATH=<RADAGENT_REPO_ROOT> \
    HF_HOME=$SCRATCH/HF_cache \
    MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
    MASTER_PORT=29500 \
    RANK=\${SLURM_PROCID} \
    LOCAL_RANK=\${SLURM_LOCALID} \
    WORLD_SIZE=\${SLURM_NTASKS} \
    NCCL_TIMEOUT=1800000 \
    python evaluation/process_generated_reports.py --output_folder <RADAGENT_RESULTS_DIR>/baselines/<CT_CHAT_BASELINE_RUN>/<OUTPUT_SUBDIR> --is_ct_chat_eval --ct_chat_predictions_path <RADAGENT_RESULTS_DIR>/baselines/<CT_CHAT_BASELINE_RUN>/<OUTPUT_SUBDIR>/<CT_CHAT_PREDICTIONS_FILE> --ct_chat_ground_truth_path <RADAGENT_RESULTS_DIR>/baselines/<CT_CHAT_BASELINE_RUN>/<OUTPUT_SUBDIR>/<CT_CHAT_GROUND_TRUTH_FILE>
    "
