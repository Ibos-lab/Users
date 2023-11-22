import os
import glob

# import numpy as np

lip = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/session_struct/lip/spikes/*_sp.h5"  # "/home/INT/losada.c/Documents/data/test/areas/lip/*"
bhv = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/session_struct/bhv/"

for area in [lip]:
    list = glob.glob(area)
    for path in list:
        with open("paths_sp_decoding.txt", mode="a") as f:
            f.write(path + "\n")
        with open("paths_bhv_decoding.txt", mode="a") as fb:
            split = os.path.normpath(path).split(os.sep)
            path = bhv + split[-1][:29] + split[-1][-11:-6] + "_bhv.h5"
            fb.write(path + "\n")
