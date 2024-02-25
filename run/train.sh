#!/bin/bash
#PBS -l select=1:ncpus=4:mem=20gb:ngpus=1:accelerator_model=a100
#PBS -l walltime=8:59:00
#PBS -A "MM_ClaimWorth"
 
set -e
 
module load Miniconda/3.1

conda activate my_env

export PYTHONPATH=$PYTHONPATH:/gpfs/project/flkar101/transformer_project

python /gpfs/project/flkar101/transformer_project/run/main.py