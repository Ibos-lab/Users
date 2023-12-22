import os
import glob

lip = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/*/*_bhv.h5"

for area in [lip]:
    list = glob.glob(area)
    for path in list:
        with open("paths_bhv.txt", mode="a") as f:
            f.write(path + "\n")
