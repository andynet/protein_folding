#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=256g
#SBATCH -c 1

conda activate py_data
~/miniconda3/envs/py_data/bin/python alphafold_training.py > alphafold_training.log

