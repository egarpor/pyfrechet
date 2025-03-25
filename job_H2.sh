#!/bin/sh
module load python/3.9.9

lscpu
python simulations_H2/H2_parallel.py 1
git add .
git commit -m "Simulations Hyperboloid block $block"
git push

# for b in {1..20}
# do
#   sbatch --time=01:25:00 -n 1 --cpus-per-task=32 --mem=40GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=$b --output="%j-H2-block$b.out" job_H2.sh
# done

# sbatch --time=01:25:00 -n 1 --cpus-per-task=32 --mem=40GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=1 --output="%j-H2-block1.out" job_H2.sh