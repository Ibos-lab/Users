import os
import glob

# import numpy as np
from ephysvibe.pipelines import plot_sp_b1, plot_sp_b2

lip = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/Session_struct/lip/*"  # "/home/INT/losada.c/Documents/data/test/areas/lip/*"
pfc = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/Session_struct/pfc/*"  # "/home/INT/losada.c/Documents/data/test/areas/pfc/*"
v4 = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/Session_struct/v4/*"  # "/home/INT/losada.c/Documents/data/test/areas/v4/*"
output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots2/"

file1 = open("paths.txt", "r")
Lines = file1.readlines()
for line in Lines:
    path = line.strip()
    o = output + "/" + os.path.normpath(path).split(os.sep)[-2] + "/" + "b2"
    # o = output + "/" + os.path.normpath(path).split(os.sep)[-2] + "/" + "b1"
    plot_sp_b2.main(path, o, e_align=1)
    # plot_sp_b1.main(path, o, in_out=1, e_align=2, cgroup="mua")


# for area in [lip, pfc, v4]:
#     list = glob.glob(area)
#     for path in list:
#         o = output + "/" + os.path.normpath(path).split(os.sep)[-2] + "/" + "b1"

#         # plot_sp_b1.main(path,o,in_out=1,e_align=2,cgroup='good')
#         # print(o)
