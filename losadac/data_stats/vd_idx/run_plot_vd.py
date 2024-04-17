import os
import glob
import platform
import sp_plot_vd

if platform.system() == "Linux":
    basepath = (
        "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/"
    )
elif platform.system() == "Windows":
    basepath = "C:/Users/camil/Documents/int/"


output = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/plots_test_vd/"


areas = ["v4", "lip", "pfc"]

for area in areas:
    neu_path = basepath + "session_struct/" + area + "/neurons/*neu.h5"
    path_list = glob.glob(neu_path)
    for path in path_list:
        o = output + "/" + "b1"
        sp_plot_vd.main(path, o)
