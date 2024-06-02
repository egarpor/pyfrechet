#!/bin/sh

module load python/3.9.9
# cd /mnt/netapp2/Store_uni/home/usc/ei/ega/jesus/pyfrechet/notebooks/
cd notebooks/
lscpu
python NY_Taxi_Data_Tuning.py
cd ..
git add .
git commit -m "NY_Taxi_Data_Tuning 1"
git push


# sbatch --time=6:00:00 --cpus-per-task=64 --mem=10GB --mail-type=END,FAIL --mail-user=edgarcia@est-econ.uc3m.es --output="slurm-NYTuning.out" job_jesus_NYTuning.sh
