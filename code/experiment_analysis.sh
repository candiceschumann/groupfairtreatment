#!/bin/bash

outer_dir1=/Volumes/easystore/Research/groupfair/maga_exp_naive
outer_dir2=/Volumes/easystore/Research/groupfair/maga_exp
echo $outer_dir1
echo $outer_dir2

echo "T"
# python3 experiment_analysis.py remap $outer_dir1/T --in_dirs $outer_dir1/T $outer_dir2/T
python3 experiment_analysis.py T $outer_dir1/T

echo "context_size"
# python3 experiment_analysis.py remap $outer_dir1/context_size --in_dirs $outer_dir1/context_size $outer_dir2/context_size
python3 experiment_analysis.py context_size $outer_dir1/context_size

echo "c"
# python3 experiment_analysis.py remap $outer_dir1/c --in_dirs $outer_dir1/c $outer_dir2/c
python3 experiment_analysis.py c $outer_dir1/c

echo "error"
# python3 experiment_analysis.py remap $outer_dir1/error --in_dirs $outer_dir1/error $outer_dir2/error
python3 experiment_analysis.py error $outer_dir1/error

echo "delta"
# python3 experiment_analysis.py remap $outer_dir1/delta --in_dirs $outer_dir1/delta $outer_dir2/delta
python3 experiment_analysis.py delta $outer_dir1/delta

echo "ratio"
# python3 experiment_analysis.py remap $outer_dir1/ratio --in_dirs $outer_dir1/ratio $outer_dir2/ratio
python3 experiment_analysis.py ratio $outer_dir1/ratio

echo "arms"
# python3 experiment_analysis.py remap $outer_dir1/arms --in_dirs $outer_dir1/arms $outer_dir2/arms
python3 experiment_analysis.py arms $outer_dir1/arms