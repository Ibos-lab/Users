import glob
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.spatial.distance import pdist
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.structures.population_data import PopulationData
import pandas as pd


# Define functions
def get_matrix_position(neu_data, matrix, ch_start):
    cluster_ch = neu_data.cluster_ch - ch_start
    matrix = matrix - matrix.min().min()
    row, col = np.where(cluster_ch == matrix)
    return row, col


def get_ch_info(neu: NeuronData, params: List[Dict]):
    res = {}
    res["nid"] = neu.get_neuron_id()
    if neu.area != "lip":
        ch_start = np.load(
            params["sessions_path"]
            + neu.date_time
            + "/Record Node 102/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/KS"
            + neu.area.upper()
            + "/channel_map.npy"
        )[0][0]
        matrix_df = pd.read_csv(
            params["ch_path"] + neu.area + "_ch_pos.csv",
            header=0,
            index_col=0,
        )
        matrix = matrix_df.values
        row, col = get_matrix_position(neu, matrix, ch_start)
    else:
        row, col = [int(neu.cluster_ch)], [0]
    res["ch_row"] = row[0]
    res["ch_col"] = col[0]
    return res


# Define parameters
areas = ["lip", "pfc", "v4"]
subject = "Riesling"
# paths
filepaths = {
    "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/lip/neurons/",
    "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/pfc/neurons/",
    "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/v4/neurons/",
}

for area in areas:
    path = filepaths[area]
    neu_path = path + "*neu.h5"
    path_list = glob.glob(neu_path)

    attr_dtype = {
        "sp_samples": np.float16,
        "cluster_ch": np.float16,
        "cluster_number": np.float16,
        "block": np.float16,
        "trial_error": np.float16,
        "sample_id": np.float16,
        "test_stimuli": np.float16,
        "test_distractor": np.float16,
        "cluster_id": np.float16,
        "cluster_depth": np.float16,
        "code_samples": np.float16,
        "position": np.float16,
        "pos_code": np.float16,
        "code_numbers": np.float16,
    }
    popu = PopulationData.get_population(path_list, attr_dtype)

    ch_path = (
        "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/units_info/data/"
    )
    sessions_path = "/envau/work/invibe/USERS/IBOS/openephys/Riesling/"
    params = {"ch_path": ch_path, "sessions_path": sessions_path}
    df_ch = popu.execute_function(get_ch_info, params=params, n_jobs=1, ret_df=True)
    df_ch.to_csv("population_ch_" + area + ".csv")
