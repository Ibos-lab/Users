import os
import glob
import platform
import compute_neurons_df
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json
import os
import argparse
from pathlib import Path
import logging
from collections import defaultdict
from typing import Dict
import pandas as pd

if platform.system() == "Linux":
    basepath = (
        "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/"
    )
elif platform.system() == "Windows":
    basepath = "C:/Users/camil/Documents/int/"


save_path = "./"
sessions_path = "/envau/work/invibe/USERS/IBOS/openephys/Riesling/"
ch_pos_path = "/envau/work/invibe/USERS/LOSADA/Users/losadac/data_stats/"
areas = ["v4", "pfc"]  # ,'pfc']
time_before = 500
start = -200
end = 1000
mov_avg_win = 100
selec_win = 75
vd_pwin = 75
vd_avg_win = 100


## main computation
area_info = {}
for area in areas:
    neu_path = basepath + "session_struct/" + area + "/neurons/*neu.h5"
    path_list = glob.glob(neu_path)
    if area == "v4":
        st_v = 50
        end_v = 200
        st_d = 100
        end_d = 300
    elif area == "pfc":
        st_v = 100
        end_v = 250
        st_d = 100
        end_d = 300
    info = Parallel(n_jobs=-1)(
        delayed(compute_neurons_df.main)(
            path_list[i],
            sessions_path=sessions_path,
            ch_pos_path=ch_pos_path,
            time_before=time_before,
            start=start,
            end=end,
            mov_avg_win=mov_avg_win,
            selec_win=selec_win,
            st_v=st_v,
            end_v=end_v,
            st_d=st_d,
            end_d=end_d,
            vd_pwin=vd_pwin,
            vd_avg_win=vd_avg_win,
        )
        for i in tqdm(range(len(path_list)))
    )
    area_info[area] = info
    df_keys = list(area_info[area][0].keys())
    df_aux: Dict[str, list] = defaultdict(list)
    for i in range(len(area_info[area])):
        for key in df_keys:
            df_aux[key] += [area_info[area][i][key]]
    areas_df = pd.DataFrame(df_aux)
    file_name = save_path + area + "_neurons_info.csv"
    areas_df.to_csv(file_name, index=False)
    parameters = {
        "script": "compute_neurons_df.py",
        "date_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "time_before": time_before,
        "start": start,
        "end": end,
        "mov_avg_win": mov_avg_win,
        "selec_win": selec_win,
        "vd_pwin": vd_pwin,
        "vd_avg_win": vd_avg_win,
        "st_v": st_v,
        "end_v": end_v,
        "st_d": st_d,
        "end_d": end_d,
    }
    with open(save_path + area + "_pipeline_parameters.json", "w") as f:
        json.dump(parameters, f)
