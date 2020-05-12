#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L-BFGS Optimization script
"""
# %%
from structure import Structure
import argparse
import os

parser = argparse.ArgumentParser('L-BFGS based Structure Optimization')

parser.add_argument('-d', '--domain', metavar='', required=True, help='Domain Name')
parser.add_argument('-n', '--numoptim', type=int, metavar='', required=False, help='Send "n" different jobs to cluster with different random states', default=0)
parser.add_argument('-s', '--structurepath', metavar='', required=False, help='path where Structure is saved. If not provided a new object is created', default='')
parser.add_argument('-r', '--randomstate', type=int, metavar='', required=False, help='Domain Name', default=1)
parser.add_argument('-o', '--outputdir', metavar='', required=False, help='Directory where the output should be saved', default='./')
parser.add_argument('-i', '--iterations', type=int, metavar='', required=False, help='Number of iterations', default=20)
parser.add_argument('-v', '--verbose', type=int, metavar='', required=False, help='How often should the program print info about losses. Default=0', default=0)

args = parser.parse_args()

if __name__ == '__main__':
    if args.numoptim == 0:
        struct = Structure(args.domain, args.randomstate)
        struct.optimize(args.iterations, args.outputdir, args.verbose)
        
    else:
        os.system(f"mkdir -p {args.outputdir}/temp_{args.domain}")
        os.system(f"mkdir {args.outputdir}/{args.domain}")
        for i in range(args.numoptim):
            with open(f'{args.outputdir}/temp_{args.domain}/{args.domain}_{i}.sh', 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('#SBATCH --mem=4g\n')
                f.write('#SBATCH -t 1000\n')
                f.write(f'#SBATCH -o ../../steps/garbage/{args.domain}-%j.out\n')
                f.write(f'python3 lbfgs_optimize.py -d {args.domain} -r {i} -o {args.outputdir}/{args.domain} -i {args.iterations} -v {args.verbose}')
            os.system(f"sbatch {args.outputdir}/temp_{args.domain}/{args.domain}_{i}.sh")