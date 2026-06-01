# Submit the first job
SCRIPT='slurm_scripts/runs_in_overleaf/training/training_part_1_v8minus.slurm'

#JOB1=$(sbatch --parsable $SCRIPT)
JOB1=1629097
# Submit the second job, depending on the first
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 $SCRIPT)

# Submit the second job, depending on the first
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 $SCRIPT)

# Submit the second job, depending on the first
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 $SCRIPT)

JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 $SCRIPT)