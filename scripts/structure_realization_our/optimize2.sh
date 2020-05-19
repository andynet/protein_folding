#!/bin/bash

python3 optimize_script.py -d $1 -n 1000 -o ../../steps/folded_structures/optimized -i 500 -lr 0.001 -ld 1.0 -m 0.5
