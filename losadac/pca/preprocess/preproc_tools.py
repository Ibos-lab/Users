from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.trials import align_trials, select_trials
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List


def to_python_hdf5(dat: List, save_path: Path):
    """Save data in hdf5 format."""
    # save the data
    with h5py.File(save_path, "w") as f:
        for i_d in range(len(dat)):
            group = f.create_group(str(i_d))

            for key, value in zip(dat[i_d].keys(), dat[i_d].values()):
                group.create_dataset(key, np.array(value).shape, data=value)
    f.close()


def select_sample_test_aligned_trials(
    neu_data, select_block, code, time_before_sample, time_before_test, error_type=0
):
    sp_sample_on, mask_s = align_trials.align_on(
        sp_samples=neu_data.sp_samples,
        code_samples=neu_data.code_samples,
        code_numbers=neu_data.code_numbers,
        trial_error=neu_data.trial_error,
        block=neu_data.block,
        pos_code=neu_data.pos_code,
        select_block=select_block,
        select_pos=code,
        event="sample_on",
        time_before=time_before_sample,
        error_type=error_type,
    )
    # Select trials aligned to sample onset
    sp_test_on, mask_t = align_trials.align_on(
        sp_samples=neu_data.sp_samples,
        code_samples=neu_data.code_samples,
        code_numbers=neu_data.code_numbers,
        trial_error=neu_data.trial_error,
        block=neu_data.block,
        pos_code=neu_data.pos_code,
        select_block=select_block,
        select_pos=code,
        event="test_on_1",
        time_before=time_before_test,
        error_type=error_type,
    )
    if np.any(mask_s != mask_t):
        return "error"
    return sp_sample_on, sp_test_on, mask_s, mask_t


def get_neuron_sample_test_fr(
    path,
    time_before_sample,
    time_before_test,
    idx_start_sample,
    idx_end_sample,
    idx_start_test,
    idx_end_test,
    n_test,
    min_trials,
    nonmatch=True,
    avgwin=50,
    n_sp_sec=5,
):
    neu_data = NeuronData.from_python_hdf5(path)
    select_block = 1
    code = 1
    # Select trials aligned to sample onset
    sp_sample_on, sp_test_on, mask_s, mask_t = select_sample_test_aligned_trials(
        neu_data, select_block, code, time_before_sample, time_before_test, error_type=0
    )
    # Build masks to select trials with match in the n_test
    mask_match = np.where(
        neu_data.test_stimuli[mask_t, n_test - 1] == neu_data.sample_id[mask_t],
        True,
        False,
    )
    mask_neu = neu_data.sample_id[mask_t] == 0
    # Build masks to select trials with the selected number of test presentations
    max_test = neu_data.test_stimuli[mask_t].shape[1]
    mask_ntest = (
        max_test - np.sum(np.isnan(neu_data.test_stimuli[mask_t]), axis=1)
    ) > (n_test - 1)

    if nonmatch:  # include nonmatch trials
        mask_match_neu = np.logical_or(mask_ntest, mask_neu)
    else:
        mask_match_neu = np.logical_or(mask_match, mask_neu)
    if np.sum(mask_match_neu) < 20:
        return {"fr": None}

    # Average fr across time
    avg_sample_on = firing_rate.moving_average(
        sp_sample_on[mask_match_neu], win=avgwin, step=1
    )[:, idx_start_sample:idx_end_sample]
    avg_test1_on = firing_rate.moving_average(
        sp_test_on[mask_match_neu], win=avgwin, step=1
    )[:, idx_start_test:idx_end_test]
    # Concatenate sample and test aligned data
    sp = np.concatenate((avg_sample_on, avg_test1_on), axis=1)
    # Check fr
    ms_fr = np.nanmean(sp) * 1000 > n_sp_sec
    if not ms_fr:
        return {"fr": None}
    # Check number of trials
    sample_id = neu_data.sample_id[mask_t][mask_match_neu]
    samples = [0, 11, 15, 55, 51]
    for s_id in samples:
        sample_fr = sp[np.where(sample_id == s_id, True, False)]
        if sample_fr.shape[0] < min_trials:
            return {"fr": None}
    # Get trials grouped by sample
    fr_samples = select_trials.get_sp_by_sample(sp, sample_id, samples=samples)

    if fr_samples is None:
        return {"fr": None}
    return {"fr": fr_samples}
