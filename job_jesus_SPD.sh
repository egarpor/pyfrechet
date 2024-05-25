#!/bin/sh

module load python/3.9.9
# cd /mnt/netapp2/Store_uni/home/usc/ei/ega/jesus/pyfrechet/simulations_SPD/
cd simulations_SPD/
lscpu
python main_parallel.py $block
cd ..
git add .
git commit -m "Simulations SPD block $block"
git push

# for b in {1..50}
# do
#   sbatch --time=4:00:00 --cpus-per-task=56 --mem=10GB --mail-type=END,FAIL --mail-user=edgarcia@est-econ.uc3m.es --export=block=$b --output="slurm-T2-block$b-jobid-%j.out" job_jesus_T2.sh
# done
