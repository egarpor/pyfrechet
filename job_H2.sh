#!/bin/sh
module load python/3.9.9

lscpu
python simulations_H2/H2_parallel.py $block
git add .
git commit -m "Simulations hyperboloid block $block"
git push

# for b in {1..20}
# do
#   sbatch --time=4:00:00 --cpus-per-task=56 --mem=10GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=$b --output="slurm-H2-block$b-jobid-%j.out" job_H2.sh
# done

# sbatch --time=0:10:00 --cpus-per-task=56 --mem=10GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=1 --output="slurm-H2-block1-jobid-%j.out" job_H2.sh
