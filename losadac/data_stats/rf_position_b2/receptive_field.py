import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from ephysvibe.structures.neuron_data import NeuronData
from scipy import stats
import pandas as pd
import glob


def get_v_resp_loc(neu, params, rf_loc=None):
    # Check rf of the neurons
    if rf_loc is not None:
        neu.check_fr_loc(rf_loc)
    nid = neu.get_neuron_id()
    # Define masks
    b1_mask = neu.block == 1
    b2_mask = neu.block == 2
    in_mask = neu.pos_code == 1
    # check b2
    if np.sum(b2_mask) == 0:
        results = {
            "nid": nid,
            "v_resp_out": np.nan,
            "x_pos_b2": np.nan,
            "y_pos_b2": np.nan,
            "x_pos_b1": np.nan,
            "y_pos_b1": np.nan,
            "code": np.nan,
            "op_code": np.nan,
            "fr": np.nan,
        }
        return results
    # get the screen position of sample in b1 in contralateral trials
    u_pos, u_count = np.unique(
        neu.position[np.logical_and(b1_mask, in_mask)], axis=0, return_counts=True
    )
    imax = np.argmax(u_count)
    x_pos_b1, y_pos_b1 = u_pos[imax][0][0], u_pos[imax][0][1]
    # Concatenate and get unique position and code during b2 trials
    pos_and_code = np.concatenate(
        [neu.position[b2_mask][:, 0], neu.pos_code[b2_mask].reshape(-1, 1)], axis=1
    )
    u_pos = np.unique(pos_and_code, axis=0)
    # Find the closest screen position to b1 in b2
    diff = abs(abs(u_pos[:, :2]) - abs(np.array([x_pos_b1, y_pos_b1])))
    idx = np.argmin(np.sum(diff, axis=1))
    x_pos_b2 = abs(u_pos[idx, 0]) * np.sign(x_pos_b1)
    y_pos_b2 = abs(u_pos[idx, 1]) * np.sign(y_pos_b1)
    idx_in = np.logical_and(u_pos[:, 0] == x_pos_b2, u_pos[:, 1] == y_pos_b2)
    # if np.sum(idx_in)==0:
    #     x_pos = get_aprox_pos(x_pos)
    #     y_pos = get_aprox_pos(y_pos)
    #     idx_in = np.logical_and(u_pos[:,0]==x_pos, u_pos[:,1]==y_pos)
    code_in = str(int(u_pos[idx_in][0][2]))
    idx_out = np.logical_and(u_pos[:, 0] == -x_pos_b2, u_pos[:, 1] == -y_pos_b2)
    code_out = str(int(u_pos[idx_out][0][2]))
    # Align spikes to target onset and get trials by code
    align_sp, alig_mask = neu.align_on(
        select_block=2, event="target_on", time_before=200, error_type=0
    )
    pos_code = neu.pos_code[alig_mask]
    sp_pos = {}
    for code in np.unique(pos_code):
        code_mask = pos_code == code
        sp_pos[str(int(code))] = align_sp[code_mask]
    # Compute t-test comparing baseline with target presentation (in vs opposite loc)
    sp = sp_pos[code_in]
    sp_op = sp_pos[code_out]
    # Check fr
    fr = np.nanmean(sp[:, :1000]) * 1000
    if fr < 1:
        results = {
            "nid": nid,
            "v_resp_out": np.nan,
            "x_pos_b2": np.nan,
            "y_pos_b2": np.nan,
            "x_pos_b1": np.nan,
            "y_pos_b1": np.nan,
            "code": np.nan,
            "op_code": np.nan,
            "fr": fr,
        }
        return results
    p = stats.ttest_rel(np.mean(sp[:, :200], axis=1), np.mean(sp[:, 200:400], axis=1))
    p = p[1] < 0.05
    p_op = stats.ttest_rel(
        np.mean(sp_op[:, :200], axis=1), np.mean(sp_op[:, 200:400], axis=1)
    )
    p_op = p_op[1] < 0.05
    v_resp_out = True
    # check loc of rf in b1
    if p == False and p_op == False:
        v_resp_out = False
    elif np.all(neu.rf_loc[b1_mask] == neu.pos_code[b1_mask]):
        if p and p_op == False:
            v_resp_out = False
    else:
        if p_op and p == False:
            v_resp_out = False
    results = {
        "nid": nid,
        "v_resp_out": v_resp_out,
        "x_pos_b2": x_pos_b2,
        "y_pos_b2": y_pos_b2,
        "x_pos_b1": x_pos_b1,
        "y_pos_b1": y_pos_b1,
        "code": code_in,
        "op_code": code_out,
        "fr": fr,
    }
    return results


def read_and_compute(path, params, rf_loc=None):
    neu = NeuronData.from_python_hdf5(path)
    neu = get_v_resp_loc(params=params, neu=neu, rf_loc=rf_loc)
    return neu


def run_rf(paths, processing, **kwargs):
    params = []
    for idict in processing:
        params.append(processing[idict])
    neu_path = paths["input_files"] + "*neu.h5"
    path_list = glob.glob(neu_path)
    rf_loc = None
    if paths["input_rf_loc"] is not None:
        rf_loc = pd.read_csv(paths["input_rf_loc"])

    res = Parallel(n_jobs=-1)(
        delayed(read_and_compute)(path, params, rf_loc) for path in tqdm(path_list)
    )

    df = pd.DataFrame(res)
    df.to_csv("rf_b2.csv")