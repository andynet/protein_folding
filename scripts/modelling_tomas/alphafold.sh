#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 10080
#SBATCH --mem=192g

python train_alphafold.py
