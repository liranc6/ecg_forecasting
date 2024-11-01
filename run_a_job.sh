#!/bin/bash
#SBATCH -A ash                                   # Account
#SBATCH -p ash                                   # Partition
#SBATCH --job-name=multi_gpu_jobs                # Job name
#SBATCH -o ./mrDiff_runs/$A/out_job_%a.txt       # stdout file with job ID (%A) and task ID (%a)
#SBATCH -e ./mrDiff_runs/%A/out_job_%a.txt       # stderr file with job ID (%A) and task ID (%a)
#SBATCH --array=0-8                              # Array range for 9 jobs (0 to 8)
#SBATCH --ntasks=1                               # One task per job
#SBATCH --gres=gpu:1                             # Request 1 GPU per job
#SBATCH --cpus-per-task=16                       # Number of CPU cores per task
#SBATCH --mem=64G                                # Increase memory per job
#SBATCH --time=04:00:00                          # Increase time limit

module purge                                      # Clean active modules list

# Create output directory with the job ID if it doesn't exist
mkdir -p ./mrDiff_runs/${SLURM_ARRAY_JOB_ID}

# Define the base command
BASE_CMD="/home/liranc6/miniconda3/envs/ecg/bin/python ./liran_project/mrdiff/main_default.py"

# Define the array of flag sets
FLAG_SETS=(
    "--emd.use_emd True --emd.num_sifts 3"
    "--emd.use_emd True --emd.num_sifts 4"
    "--emd.use_emd True --emd.num_sifts 5"
    "--emd.use_emd False --training.smoothing.smoothed_factors '[11_31_41]'"
    "--emd.use_emd False --training.smoothing.smoothed_factors '[11_31_41_51]'"
    "--emd.use_emd False --training.smoothing.smoothed_factors '[31_41_51]'"
    "--emd.use_emd False --training.smoothing.smoothed_factors '[31_41_51_61]'"
    "--emd.use_emd False --training.smoothing.smoothed_factors '[11_31_41_51_61]'"
    "--emd.use_emd False --training.smoothing.smoothed_factors '[1]'"
)

# Select the flag set for this job
FLAGS="${FLAG_SETS[${SLURM_ARRAY_TASK_ID}]}"

# Run the command with the selected flags
eval "$BASE_CMD $FLAGS"
