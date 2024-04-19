import os
import glob
import platform
import sp_plot_vd
from joblib import Parallel, delayed
from tqdm import tqdm

if platform.system() == "Linux":
    basepath = (
        "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/"
    )
elif platform.system() == "Windows":
    basepath = "C:/Users/camil/Documents/int/"


output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/plots_test_vd/"

o = output + "/" + "b1"
areas = ["pfc", "lip"]  # "v4",

for area in areas:
    neu_path = basepath + "session_struct/" + area + "/neurons/*neu.h5"
    path_list = glob.glob(neu_path)
    Parallel(n_jobs=-1)(
        delayed(sp_plot_vd.main)(path_list[i], o) for i in tqdm(range(len(path_list)))
    )
