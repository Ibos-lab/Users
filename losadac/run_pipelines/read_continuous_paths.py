import os
import glob

# import numpy as np
from ephysvibe.pipelines import plot_sp_b1

lip = "/envau/work/invibe/USERS/IBOS/openephys/Riesling/*/*/*/*/*/*/continuous.dat"  # "/home/INT/losada.c/Documents/data/test/areas/lip/*"

for area in [lip]:
    list = glob.glob(area)
    for path in list:
        with open("paths_continuous.txt", mode="a") as f:
            f.write(path + "\n")
