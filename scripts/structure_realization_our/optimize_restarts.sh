#!/bin/bash

python3 optimize_restarts.py -d $1 -dp ../../steps/test_predictions/$1.pred.pt -i 200 -lr 0.05 -m 0.5 -o ../../steps/folded_structures/restarted_optim -n 1000
