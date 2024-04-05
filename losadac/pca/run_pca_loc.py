from ephysvibe.structures.neuron_data import NeuronData
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import glob
from ephysvibe.trials import align_trials
from ephysvibe.trials.spikes import firing_rate
import platform
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
from pathlib import Path
from typing import Dict, List
import pca_tools
from scipy import stats

seed = 2024


def to_python_hdf5(dat: List, save_path: Path):
    """Save data in hdf5 format."""
    # save the data
    with h5py.File(save_path, "w") as f:
        for i_d in range(len(dat)):
            group = f.create_group(str(i_d))

            for key, value in zip(dat[i_d].keys(), dat[i_d].values()):
                group.create_dataset(key, np.array(value).shape, data=value)
    f.close()


def get_fr_samples(sp, sample_id, start, end, samples, min_trials):
    # parameters
    win = 50
    # Check fr
    ms_fr = np.nanmean(sp[:, start:end]) * 1000 > 5
    if not ms_fr:
        return None
    # Average spikes
    avg_sample = firing_rate.moving_average(sp, win=win, step=1)
    fr_samples = []
    for s_id in samples:
        sample_fr = avg_sample[np.where(sample_id == s_id, True, False), start:end]
        # Check number of trials
        if sample_fr.shape[0] < min_trials:
            return None
        fr_samples.append(np.mean(sample_fr, axis=0))
    return fr_samples


def select_trials(neu_data, select_block, code, time_before, error_type=0):
    sp_sample_on, mask_sample = align_trials.align_on(
        sp_samples=neu_data.sp_samples,
        code_samples=neu_data.code_samples,
        code_numbers=neu_data.code_numbers,
        trial_error=neu_data.trial_error,
        block=neu_data.block,
        pos_code=neu_data.pos_code,
        select_block=select_block,
        select_pos=code,
        event="sample_on",
        time_before=time_before,
        error_type=error_type,
    )
    sp_test1_on, mask_test1 = align_trials.align_on(
        sp_samples=neu_data.sp_samples,
        code_samples=neu_data.code_samples,
        code_numbers=neu_data.code_numbers,
        trial_error=neu_data.trial_error,
        block=neu_data.block,
        pos_code=neu_data.pos_code,
        select_block=select_block,
        select_pos=code,
        event="test_on_1",
        time_before=200,
        error_type=error_type,
    )
    if np.any(mask_sample != mask_test1):
        return "error"
    return sp_sample_on, sp_test1_on, mask_sample, mask_test1


def get_neuron_sample_test1_fr(
    path, time_before, start, end, end_test, n_test, min_trials, nonmatch=True
):
    neu_data = NeuronData.from_python_hdf5(path)

    position = neu_data.position[
        np.logical_and(neu_data.block == 1, neu_data.pos_code == 1)
    ]
    u_pos = np.unique(position, axis=0)

    if u_pos.shape[0] > 1:
        print("Position of the sample change during the session %s" % path)
        return {"fr": None}
    if np.logical_or(u_pos[0][0][0] != 5, u_pos[0][0][1] != 5):
        return {"fr": None}

    select_block = 1
    code = 1
    idx_start = time_before + start
    idx_end = time_before + end
    # Select trials aligned to sample onset
    sp_sample_on, sp_test1_on, mask_sample, mask_test1 = select_trials(
        neu_data, select_block, code, time_before, error_type=0
    )

    mask_match = np.where(
        neu_data.test_stimuli[mask_test1, n_test - 1] == neu_data.sample_id[mask_test1],
        True,
        False,
    )
    mask_neu = neu_data.sample_id[mask_test1] == 0
    max_test = neu_data.test_stimuli[mask_test1].shape[1]
    mask_ntest = (
        max_test - np.sum(np.isnan(neu_data.test_stimuli[mask_test1]), axis=1)
    ) > (n_test - 1)
    # mask_match_neu = np.logical_or(mask_match,mask_neu)
    # mask_match_neu = np.logical_or(np.full(mask_neu.shape,True),mask_neu)
    if nonmatch:
        mask_match_neu = np.logical_or(mask_ntest, mask_neu)
    else:
        mask_match_neu = np.logical_or(mask_match, mask_neu)

    if np.sum(mask_match_neu) < 20:
        return {"fr": None}
    sp = np.concatenate(
        (
            sp_sample_on[mask_match_neu, : time_before + 450 + 200],
            sp_test1_on[mask_match_neu, : end_test + 400],
        ),
        axis=1,
    )
    sample_id = neu_data.sample_id[mask_test1][mask_match_neu]
    fr_samples = get_fr_samples(
        sp,
        sample_id,
        start=idx_start,
        end=idx_end,
        samples=[0, 11, 15, 55, 51],
        min_trials=min_trials,
    )
    if fr_samples is None:
        return {"fr": None}
    return {"fr": fr_samples}


if platform.system() == "Linux":
    basepath = (
        "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/"
    )
elif platform.system() == "Windows":
    basepath = "C:/Users/camil/Documents/int/"

area = "v4"
neu_path = basepath + "/session_struct/" + area + "/neurons/*neu.h5"
path_list = glob.glob(neu_path)

# Load data
n_test = 1
nonmatch = True
min_trials = 15
time_before = 500
start = -200
end_test = n_test * 450 + 200
end = 450 + 200 + 200 + end_test

idx_start = time_before + start
idx_end = time_before + end

data = Parallel(n_jobs=-1)(
    delayed(get_neuron_sample_test1_fr)(
        path, time_before, start, end, end_test, n_test, min_trials, nonmatch
    )
    for path in tqdm(path_list)
)

neurons_fr = []
for asc in data:
    fr = asc["fr"]
    if fr is not None:
        neurons_fr.append(asc)
basepath = (
    "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/pca/data/" + area + "/"
)
to_python_hdf5(
    dat=neurons_fr,
    save_path=basepath
    + area
    + "_pos5-5"
    + "_win50_test"
    + str(n_test)
    + "_wnonmatch_min"
    + str(min_trials)
    + "tr_pca.h5",
)
