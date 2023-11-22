import os
import glob

# import numpy as np
from ephysvibe.pipelines import generalized_phase


output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots2/generalized_phase/median_ch_substracted"

file1 = open("paths_continuous.txt", "r")
Lines = file1.readlines()
for line in Lines:
    path = line.strip()

    generalized_phase.main(path, output)
