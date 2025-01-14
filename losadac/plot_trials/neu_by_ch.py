import pandas as pd
import platform
import glob
import os
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.dataviz import plot_raster
import matplotlib.pyplot as plt
import numpy as np

basepath = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure"
pathcsv = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots_by_ch"
sessions_path = "/envau/work/invibe/USERS/IBOS/openephys/Riesling/"

for area in ["pfc", "v4"]:
    subpath = f"/session_struct/{area}/neurons"
    path_list = glob.glob(f"{basepath}{subpath}/*")
    lenpaths = len(path_list)
    print(f"{area}: {lenpaths}")
    for n, path_n in enumerate(path_list):

        print(f"{area}: {n}/{lenpaths}")
        neu_n = NeuronData.from_python_hdf5(path_n)
        nid = neu_n.get_neuron_id()
        sp, conv = plot_raster.prepare_data_plotb1(
            neu_n, rf_stim_loc=["contra", "ipsi"], cerotr=False, percentile=False
        )

        session = neu_n.date_time
        ch_start = np.load(
            sessions_path
            + session
            + "/Record Node 102/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/KS"
            + area.upper()
            + "/channel_map.npy"
        )[0][0]
        cluster_ch = neu_n.cluster_ch - ch_start

        path1 = f"{pathcsv}/all_trials/{area}/{cluster_ch}"

        if not os.path.exists(path1):
            os.makedirs(path1)

        fig1 = plot_raster.plot_sp_b1(neu_n, sp, conv)
        fig1.savefig(path1 + "/" + nid + ".jpg", format="jpg")
        plt.close("all")
