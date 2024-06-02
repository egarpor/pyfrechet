#!/bin/sh

module load python/3.9.9
# cd /mnt/netapp2/Store_uni/home/usc/ei/ega/jesus/pyfrechet/notebooks/
cd notebooks/
lscpu
python NY Taxi Data Tuning.py
cd ..
git add .
git commit -m "NY Taxi Data Tuning 1"
git push


# sbatch --time=5:00:00 --cpus-per-task=64 --mem=10GB --mail-type=BEGIN,END,FAIL --mail-user=edgarcia@est-econ.uc3m.es --output="slurm-NYTuning-jobid.out" job_jesus_NYTuning.sh
