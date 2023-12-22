import os
import glob

file_path = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/*/*/*_neu.h5"
for area in [file_path]:
    list = glob.glob(area)
    for path in list:
        with open("paths_neurons.txt", mode="a") as f:
            f.write(path + "\n")
