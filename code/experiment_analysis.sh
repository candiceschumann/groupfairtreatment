#!/bin/bash


echo "T"
python3 experiment_analysis.py remap ../maga_exp_naive/T --in_dirs ../maga_exp_naive/T ../maga_exp/T
python3 experiment_analysis.py T ../maga_exp_naive/T

# echo "context_size"
# python3 experiment_analysis.py remap ../maga_exp_naive/context_size --in_dirs ../maga_exp_naive/context_size ../maga_exp/context_size
# python3 experiment_analysis.py context_size ../maga_exp_naive/context_size

# echo "c"
# python3 experiment_analysis.py remap ../maga_exp_naive/c --in_dirs ../maga_exp_naive/c ../maga_exp/c
# python3 experiment_analysis.py c ../maga_exp_naive/c

# echo "error"
# python3 experiment_analysis.py remap ../maga_exp_naive/error --in_dirs ../maga_exp_naive/error ../maga_exp/error
# python3 experiment_analysis.py error ../maga_exp_naive/error

# echo "delta"
# python3 experiment_analysis.py remap ../maga_exp_naive/delta --in_dirs ../maga_exp_naive/delta ../maga_exp/delta
# python3 experiment_analysis.py delta ../maga_exp_naive/delta

# echo "ratio"
# python3 experiment_analysis.py remap ../maga_exp_naive/ratio --in_dirs ../maga_exp_naive/ratio ../maga_exp/ratio
# python3 experiment_analysis.py ratio ../maga_exp_naive/ratio

# echo "arms"
# python3 experiment_analysis.py remap ../maga_exp_naive/arms --in_dirs ../maga_exp_naive/arms ../maga_exp/arms
# python3 experiment_analysis.py arms ../maga_exp_naive/arms