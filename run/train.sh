#!/bin/bash
#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1:accelerator_model=rtx8000
#PBS -l walltime=14:59:00
#PBS -A "MM_ClaimWorth"
 
set -e
 
module load Miniconda/3.1

conda activate my_env

python /gpfs/project/flkar101/