import os
import glob
import logging

# import numpy as np
from ephysvibe.pipelines.preprocessing import compute_eye


output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure"
file1 = open(
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/run_pipelines/paths_continuous_bhv.txt",
    "r",
)
file2 = open(
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/run_pipelines/paths_bhv.txt",
    "r",
)
Lines = file1.readlines()
Lines2 = file2.readlines()
file1.close()
file2.close()
for line, line2 in zip(Lines, Lines2):
    ks_path = line.strip()
    bhv_path = line2.strip()
    print(ks_path)

    try:
        compute_eye.main(ks_path, bhv_path, output)
    except FileExistsError:
        logging.error("path does not exist")
        continue
