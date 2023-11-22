#!/bin/bash


#SBATCH -J run_compute_trials
#SBATCH -p batch
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:06:00
#SBATCH -N 1

filename='/envau/work/invibe/USERS/IBOS/code/flow/paths.txt'
n=1
while read line; do
# reading each line
echo $n $line
python -m ephysvibe.pipelines.plot_sp_b1 "$line" -o "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots2/"
n=$((n+1))
done < $filename