#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=ReLIC_
#SBATCH --output=job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --gpus=geforce_gtx_1080_ti:1
#SBATCH --qos=batch

# Activate everything you need
module load cuda/11.2
pyenv activate venv
# Run your python code
python simclr.py --mode relic --filename relic4