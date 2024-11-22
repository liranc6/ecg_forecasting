#!/bin/bash
# run with the command: sbatch run_a_job.sh

# #SBATCH -w nlp-ada-1
#SBATCH -A ash
#SBATCH -p ash
#SBATCH --job-name=exp                          # Job name
#SBATCH -o ./mrDiff_runs/%A/out_job_%a.txt       # stdout file with job ID (%A) and task ID (%a)
#SBATCH -e ./mrDiff_runs/%A/out_job_%a.txt       # stderr file with job ID (%A) and task ID (%a)
#SBATCH --array=0                              # Array range for 9 jobs (0 to 8)
#SBATCH --gres=gpu:3                             # Request 1 GPU per job
#SBATCH --cpus-per-task=96                       # Number of CPU cores per task

module purge                                      # Clean active modules list

# Ensure Conda initialization
eval "$(conda shell.bash hook)"
conda init
conda activate ecg

# Create output directory with the job ID if it doesn't exist
mkdir -p ./mrDiff_runs/${SLURM_ARRAY_JOB_ID}

# Define the base command
BASE_CMD="srun ./liran_project/mrdiff/main_default.py"

# "--emd.use_emd True --emd.num_sifts 3"
#     "--emd.use_emd True --emd.num_sifts 4"
#     "--emd.use_emd True --emd.num_sifts 5"
#     "--emd.use_emd False --training.smoothing.smoothed_factors '[11_31_41]'"
#     "--emd.use_emd False --training.smoothing.smoothed_factors '[11_31_41_51]'"
#     "--emd.use_emd False --training.smoothing.smoothed_factors '[31_41_51]'"
#     "--emd.use_emd False --training.smoothing.smoothed_factors '[31_41_51_61]'"
#     "--emd.use_emd False --training.smoothing.smoothed_factors '[11_31_41_51_61]'"
#     "--emd.use_emd False --training.smoothing.smoothed_factors '[1]'"

# Define the array of flag sets
FLAG_SETS=(
    ""
)

# Select the flag set for this job
FLAGS="${FLAG_SETS[${SLURM_ARRAY_TASK_ID}]}"

# Run the command with the selected flags
eval "$BASE_CMD $FLAGS"#!/bin/bash