#!/bin/sh
module load python/3.9.9

lscpu
python simulations_euc/sklearn_main_parallel.py $block
git add .
git commit -m "Simulations Euclidean block $block"
git push

#for b in {1..4}
#do
#  sbatch --time=00:45:00 -n 1 --cpus-per-task=32 --mem=8GB --mail-type=END,FAIL --mail-user=edgarcia@est-econ.uc3m.es --export=block=$b --output="%j-euc-block$b.out" job_euc.sh
#done

# sbatch --time=00:45:00 -n 1 --cpus-per-task=32 --mem=8GB --mail-type=END,FAIL --mail-user=edgarcia@est-econ.uc3m.es --export=block=1 --output="%j-euc-block1.out" job_euc.sh