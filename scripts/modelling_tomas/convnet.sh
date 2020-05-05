#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 5760
#SBATCH --mem=192g

python train_convnet.py
