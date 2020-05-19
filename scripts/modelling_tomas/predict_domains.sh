#!/bin/bash
#SBATCH -t 1
#SBATCH --mem=192g

python train_inception.py
