# import numpy as np
from ephysvibe.pipelines import plot_sp_b1, plot_sp_b2, vm_idx_b1, vm_idx_b2

import logging

# output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots2/b2"
# file1 = open("paths_sp.txt", "r")
# Lines = file1.readlines()
# for line in Lines:
#     path = line.strip()
#     print(path)
#     plot_sp_b2.main(path, output, e_align="target_on", t_before=500)

# output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots2/b1/test"
# file1 = open("paths_sp.txt", "r")
# lines = file1.readlines()
# file1 = open("paths_bhv_all.txt", "r")
# lines_bhv = file1.readlines()
# for line, line_bhv in zip(lines, lines_bhv):
#     path = line.strip()
#     path_bhv = line_bhv.strip()

#     try:
#         plot_sp_b1.main(path, path_bhv, output, e_align="sample_on", t_before=200)
#     except:
#         continue


file1 = open("paths_sp_lip.txt", "r")
Lines = file1.readlines()
file1.close()
for line in Lines:
    output = "/home/INT/losada.c/Documents/data/Riesling/plots2/detect_rf/test"
    path = line.strip()
    print(path)
    try:
        vm_idx_b2.main(path, output, e_align="target_on", t_before=200)
    except:
        continue

# output = "/home/INT/losada.c/Documents/data/Riesling/plots2/detect_rf/test"
# file1 = open("paths_sp_lip.txt", "r")
# lines = file1.readlines()
# file1 = open("paths_bhv_lip.txt", "r")
# lines_bhv = file1.readlines()

# for line, line_bhv in zip(lines, lines_bhv):
#     path = line.strip()
#     bhv_path = line_bhv.strip()
#     print(path)
#     try:
#         vm_idx_b1.main(path, bhv_path, output, e_align="target_on", t_before=200)
#     except:
#         continue
