import os
import glob

# import numpy as np
from ephysvibe.pipelines import plot_sp_b1

lip = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/session_struct/*/*/*_sp.h5"  # "/home/INT/losada.c/Documents/data/test/areas/lip/*"

for area in [lip]:
    list = glob.glob(area)
    for path in list:
        with open("paths_sp_lip.txt", mode="a") as f:
            f.write(path + "\n")
