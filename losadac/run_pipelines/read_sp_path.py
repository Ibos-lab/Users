import os
import glob

lip = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/*/*/*_sp.h5"  # "/home/INT/losada.c/Documents/data/test/areas/lip/*"

for area in [lip]:
    list = glob.glob(area)
    for path in list:
        with open("paths_sp.txt", mode="a") as f:
            f.write(path + "\n")
