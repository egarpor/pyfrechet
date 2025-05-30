#!/bin/sh
module load cesga/system miniconda3
conda activate $STORE/.conda/envs/pballs
lscpu
python simulations_sphere/sphere_parallel.py $block
git add .
git commit -m "Simulations Sphere block $block"
git push

# for b in {1..20}
# do
#   sbatch --time=01:22:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=$b --output="%j-sphere-block$b.out" conda_job_sphere.sh
# done

# sbatch --time=00:05:00 -n 1 --cpus-per-task=32 --mem=16GB --mail-type=BEGIN,END,FAIL --mail-user=dieserra@est-econ.uc3m.es --export=block=1 --output="%j-sphere-block1.out" conda_job_sphere.sh