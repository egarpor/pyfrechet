#!/bin/sh
module load python/3.9.9

lscpu
python simulations_sphere/sphere_parallel.py $block
git add .
git commit -m "Simulations Sphere block $block"
git push

#for b in {1..20}
#do
#  sbatch --time=02:00:00 --cpus-per-task=32 --mem-per-core=8GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=$b --output="slurm-sphere-block$b-jobid-%j.out" job_sphere.sh
#done

# sbatch --time=02:00:00 --cpus-per-task=32 --mem-per-core=8GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=1 --output="slurm-sphere-block1-jobid-%j.out" job_sphere.sh
