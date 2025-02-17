#!/bin/bash

#for block in {1..75}; do
#    /Users/Diego/miniconda3/envs/pballs/bin/python /Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_euc/conformal_euc_main_parallel.py "$block"
#done

for block in {1..777}; do
    /Users/Diego/miniconda3/envs/pballs/bin/python /Users/Diego/Desktop/Codigo/repo_edu_pyfrechet/pyfrechet/simulations_euc/euc_main_parallel.py "$block"
done
