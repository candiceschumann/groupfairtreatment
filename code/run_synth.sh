#!/bin/bash

for run_number in 1 2 3 4 5 6 7 8 9 10
do
    for experiment_type in T context_size c error ratio delta arms
    do
        echo $experiment_type
        python3 run_synth.py $experiment_type $run_number
    done
done
