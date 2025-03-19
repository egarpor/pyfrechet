#!/bin/sh
module load python/3.9.9
cd simulations_sphere/
lscpu
python sphere_parallel.py $block
cd ..
git add .
git commit -m "Simulations sphere block $block"
git push

# for b in {1..50}
# do
#   sbatch --time=4:00:00 --cpus-per-task=56 --mem=10GB --mail-type=END,FAIL --mail-user=edgarcia@est-econ.uc3m.es --export=block=$b --output="slurm-sphere-block$b-jobid-%j.out" job_sphere.sh
# done