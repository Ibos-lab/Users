import os
import glob


lip = "/envau/work/invibe/USERS/IBOS/openephys/Riesling/*/*/*/*/*Riesling.mat"  # "/home/INT/losada.c/Documents/data/test/areas/lip/*"


for area in [lip]:
    list = glob.glob(area)
    for path in list:
        with open("paths_bhv_mat.txt", mode="a") as f:
            f.write(path + "\n")
