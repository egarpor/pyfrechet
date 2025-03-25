#!/bin/sh
module load python/3.9.9
lscpu
python simulations_SPD/ALL_main_parallel.py $block
git add .
git commit -m "Simulations SPD block $block"
git push

# for b in {1..50}
# do
#   sbatch --time=05:00:00 -n 1 --cpus-per-task=32 --mem=15GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=$b --output="j-SPD-block$b.out" job_SPD.sh
# done

# sbatch --time=05:00:00 -n 1 --cpus-per-task=32 --mem=15GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=1 --output="j-SPD-block1.out" job_SPD.sh