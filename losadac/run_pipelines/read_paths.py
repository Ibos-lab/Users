import os
import glob

# import numpy as np
from ephysvibe.pipelines import plot_sp_b1

lip = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/session_struct/Riesling/lip/not_sure/*"  # "/home/INT/losada.c/Documents/data/test/areas/lip/*"
pfc = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/session_struct/Riesling/pfc/*"  # "/home/INT/losada.c/Documents/data/test/areas/pfc/*"
v4 = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/session_struct/Riesling/v4/*"  # "/home/INT/losada.c/Documents/data/test/areas/v4/*"

for area in [lip, pfc, v4]:
    list = glob.glob(area)
    for path in list:
        with open("paths.txt", mode="a") as f:
            f.write(path + "\n")
