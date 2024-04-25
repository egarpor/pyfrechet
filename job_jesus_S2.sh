#!/bin/sh

module load python/3.9.9
# cd /mnt/netapp2/Store_uni/home/usc/ei/ega/jesus/pyfrechet/simulations_S2/
cd simulations_S2/
lscpu
python main_parallel.py $block
cd ..
git add .
git commit -m "Simulations S2 block $block"
git push

# for b in {1..60}
# do
#   sbatch --time=3:00:00 --cpus-per-task=50 --mem=10GB --mail-type=END,FAIL --mail-user=edgarcia@est-econ.uc3m.es --export=block=$b --output="slurm-S2-block$b-jobid-%j.out" job_jesus_S2.sh
# done

