#!/bin/sh
module load python/3.9.9
lscpu
python simulations_sphere/sphere_parallel.py $block
git add .
git commit -m "Simulations Sphere block $block"
git push

# for b in {3..20}
# do
#   sbatch --time=01:22:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=$b --output="%j-sphere-block$b.out" job_sphere.sh
# done

# sbatch --time=01:22:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=1 --output="%j-sphere-block1.out" job_sphere.sh
