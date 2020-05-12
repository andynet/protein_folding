#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:42:19 2020

@author: tomasla
"""
from optimize import optimize
import argparse
import os
import time
# %% ArgParse

parser = argparse.ArgumentParser('Gradient Descent based Structure Optimization')
parser.add_argument('-d', '--domain', metavar='', required=True, help='Domain Name')
parser.add_argument('-n', '--numoptim', type=int, metavar='', required=False, help='Send "n" different jobs to cluster with different random states', default=0)
parser.add_argument('-s', '--structurepath', metavar='', required=False, help='path where Structure is saved. If not provided a new object is created', default='')
parser.add_argument('-r', '--randomstate', type=int, metavar='', required=False, help='Domain Name', default=1)
parser.add_argument('-o', '--outputdir', metavar='', required=False, help='Directory where the output should be saved', default='./')
parser.add_argument('-i', '--iterations', type=int, metavar='', required=False, help='Number of iterations', default=100)
parser.add_argument('-lr', '--learningrate', type=float, metavar='', required=False, help='Learning rate', default=1.0)
parser.add_argument('-ld', '--lrdecay', type=float, metavar='', required=False, help='Learning rate decay parameter', default=1.0)
parser.add_argument('-f', '--decayfrequency', type=int, metavar='', required=False, help='Learning rate Decay frequency', default=10)
parser.add_argument('-m', '--momentum', type=float, metavar='', required=False, help='momentum parameter', default=0.0)
parser.add_argument('-nm', '--nesterov', metavar='', required=False, help='Nesterov Momentum', default='False')
parser.add_argument('-v', '--verbose', type=int, metavar='', required=False, help='How often should the program print info about losses. Default=iterations/20', default=0)

args = parser.parse_args()

# %%
if __name__ == '__main__':
    if args.numoptim == 0:
        optimize(domain=args.domain, 
                 structure_path=args.structurepath, 
                 random_state=args.randomstate, 
                 output_dir=args.outputdir,
                 iterations=args.iterations,
                 lr=args.learningrate,
                 lr_decay=args.lrdecay,
                 decay_frequency=args.lrdecay,
                 momentum=args.momentum,
                 nesterov=args.nesterov,
                 verbose=args.verbose
                 )
    else:
        os.system(f"mkdir -p {args.outputdir}/temp_{args.domain}")
        os.system(f"mkdir {args.outputdir}/{args.domain}")
        for i in range(args.numoptim):
            with open(f'{args.outputdir}/temp_{args.domain}/{args.domain}_{i}.sh', 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('#SBATCH --mem=4g\n')
                f.write('#SBATCH -t 300\n')
                f.write(f'python3 optimize_script.py -d {args.domain} -r {i} -o {args.outputdir}/{args.domain} -i {args.iterations} -lr {args.learningrate} -ld {args.lrdecay} -f {args.decayfrequency} -m {args.momentum} -nm {args.nesterov} -v {args.verbose}')
            os.system(f"sbatch {args.outputdir}/temp_{args.domain}/{args.domain}_{i}.sh")
                
