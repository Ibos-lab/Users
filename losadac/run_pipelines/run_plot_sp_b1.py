import os
import glob

# import numpy as np
from ephysvibe.pipelines import plot_sp_b1, plot_sp_b2


output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/plots_test/"

file1 = open(
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/run_pipelines/paths_neurons.txt", "r"
)
Lines = file1.readlines()
file1.close()
for line in Lines:
    path = line.strip()
    o = output + "/" + "b1"
    plot_sp_b1.main(path, o)


# for area in [lip, pfc, v4]:
#     list = glob.glob(area)
#     for path in list:
#         o = output + "/" + os.path.normpath(path).split(os.sep)[-2] + "/" + "b1"

#         # plot_sp_b1.main(path,o,in_out=1,e_align=2,cgroup='good')
#         # print(o)
