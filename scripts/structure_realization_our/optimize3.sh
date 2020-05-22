#!/bin/bash

python3 optimize_script.py -d $1 -n 1000 -o ../../steps/folded_structures/optimized/normal_distograms -i 500 -lr 0.005 -ld 0.1 -f 200 -m 0.5
