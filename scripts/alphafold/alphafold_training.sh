#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=384g
#SBATCH --cpus-per-task=1
#SBATCH --time=1440

# conda activate py_data
~/miniconda3/envs/py_data/bin/python -u alphafold_training.py > alphafold_training.log

