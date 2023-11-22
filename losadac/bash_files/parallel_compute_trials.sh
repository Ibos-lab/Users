#! /bin/bash

parallel -a /home/INT/losada.c/Documents/codes/EphysVibe/files.txt --verbose --link --tagstring="{1/.}" -j 1 --delay 1 --memfree 6G \
    python -m ephysvibe.pipelines.compute_trials {1} {2} {3} ::: -o ::: /envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys 