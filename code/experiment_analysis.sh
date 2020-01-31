#!/bin/bash

for direct in ../maga_exp
do

    python3 experiment_analysis.py remap $direct/T
    python3 experiment_analysis.py T $direct/T

    python3 experiment_analysis.py remap $direct/context_size
    python3 experiment_analysis.py context_size $direct/context_size

    python3 experiment_analysis.py remap $direct/c
    python3 experiment_analysis.py c $direct/c

    python3 experiment_analysis.py remap $direct/error
    python3 experiment_analysis.py error $direct/error

    python3 experiment_analysis.py remap $direct/delta
    python3 experiment_analysis.py delta $direct/delta

    python3 experiment_analysis.py remap $direct/ratio
    python3 experiment_analysis.py ratio $direct/ratio

    python3 experiment_analysis.py remap $direct/arms
    python3 experiment_analysis.py arms $direct/arms
done