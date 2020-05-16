#!/bin/bash

python3 optimize_script.py -d $1 -n 1000 -o ../../steps/folded_structures/optimized -i 500 -lr 0.1 -ld 0.5 -m 0.5
