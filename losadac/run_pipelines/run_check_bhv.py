import os
import glob
import logging

# import numpy as np
from ephysvibe.pipelines.preprocessing import check_bhv


output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure"
file1 = open(
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/run_pipelines/paths_bhv_mat.txt", "r"
)
file2 = open(
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/run_pipelines/paths_continuous.txt",
    "r",
)
Lines = file1.readlines()
Lines2 = file2.readlines()
for line, line2 in zip(Lines, Lines2):
    path = line.strip()
    path_continuous = line2.strip()
    print(path)

    try:
        check_bhv.main(path, path_continuous, output)
    except FileExistsError:
        logging.error("path does not exist")
        continue
