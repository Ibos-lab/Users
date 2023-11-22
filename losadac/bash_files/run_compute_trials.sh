#!/bin/bash


#SBATCH -J run_compute_trials
#SBATCH -p batch
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:06:00
#SBATCH -N 1

filename='/envau/work/invibe/USERS/IBOS/code/EphysVibe/files.txt'
n=1
while read line; do
# reading each line
echo $n $line
python -m ephysvibe.pipelines.compute_trials "$line" -o "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys"
n=$((n+1))
done < $filename