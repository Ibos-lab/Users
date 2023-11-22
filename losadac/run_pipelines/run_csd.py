import os
import glob

# import numpy as np
from ephysvibe.pipelines import csd, laminar_power


output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots2/laminar_power/b2/"

file1 = open("paths.txt", "r")
Lines = file1.readlines()
for line in Lines:
    path = line.strip()

    laminar_power.main(path, output, block=2)
