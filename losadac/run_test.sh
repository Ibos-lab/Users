#!/bin/bash
#SBATCH --job-name=decode
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=camila.losada-gomez@univ-amu.fr
#SBATCH --mail-type=BEGIN,END,FAIL
module purge
module load all
module load anaconda
conda activate ephys
python /envau/work/invibe/USERS/LOSADA/hello.py
