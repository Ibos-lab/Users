import os
import glob
import logging

# import numpy as np
from ephysvibe.pipelines.preprocessing import check_bhv


output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure"
file1 = open(
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/run_pipelines/paths_bhv_mat.txt", "r"
)
Lines = file1.readlines()
for line in Lines:
    path = line.strip()
    print(path)

    try:
        check_bhv.main(path, output)
    except FileExistsError:
        logging.error("path does not exist")
        continue
