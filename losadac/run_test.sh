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
source activate ephys
python /envau/work/invibe/USERS/LOSADA/Users/losadac/__main__.py -m multiruns=decoding_in,decoding_out pipelines.area='v4','pfc','lip' pipelines.preprocessing.to_decode='sampleid','neutral','color','orient'