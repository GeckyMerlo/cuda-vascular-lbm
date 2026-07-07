#!/bin/bash -l
#SBATCH --job-name=HESP_Project
#SBATCH --nodes=1 
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:10:00
#SBATCH --export=NONE

set -euo pipefail

unset SLURM_EXPORT_ENV

module load cuda

timestamp=$(date +"%Y%m%d_%H%M%S")
archive_name="simulation_output_${timestamp}.zip"

make clean-output
make run

if [[ ! -d output ]]; then
    echo "Expected simulation output directory 'output' was not created." >&2
    exit 1
fi

zip -r "${archive_name}" output

echo "Created ${archive_name}"
echo "Done."