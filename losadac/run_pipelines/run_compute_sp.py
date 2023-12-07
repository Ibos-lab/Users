import os
import glob

# import numpy as np
from ephysvibe.pipelines.preprocessing import compute_spikes


output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure"

file1 = open(
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/run_pipelines/paths_continuous.txt",
    "r",
)
Lines = file1.readlines()
for line in Lines:
    path = line.strip()
    print(path)

    try:
        compute_spikes.main(
            ks_path=path,
            output_dir=output,
            areas=None,
        )
    except:
        print("error")
        continue
