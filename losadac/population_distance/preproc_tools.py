from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials import align_trials, select_trials, firing_rate
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List


def select_trials_by_percentile(x: np.ndarray, mask: np.ndarray = None):
    ntr = x.shape[0]
    if mask is None:
        mask = np.full(ntr, True)

    mntr = x[mask].shape[0]

    if mntr < 2:
        return np.full(ntr, True)
    mean_trs = np.mean(x, axis=1)

    q25, q75 = np.percentile(mean_trs[mask], [25, 75])
    iqr = q75 - q25
    upper_limit = q75 + 1.5 * iqr
    lower_limit = q25 - 1.5 * iqr

    q1mask = mean_trs > lower_limit
    q2mask = mean_trs < upper_limit

    qmask = np.logical_and(q1mask, q2mask)
    return qmask


def check_trials(x, cerotr, percentile):
    masknocero = np.full(x.shape[0], True)
    maskper = np.full(x.shape[0], True)
    if cerotr:
        masknocero = np.sum(x, axis=1) != 0
    if percentile:
        maskper = select_trials_by_percentile(x, masknocero)
    mask = np.logical_and(masknocero, maskper)
    if np.sum(mask) < 5:
        mask = np.full(x.shape[0], True)
    return mask


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


def get_fr_by_sample(
    neu,
    time_before_son: str,
    time_before_t1on: str,
    sp_son: str,
    sp_t1on: str,
    mask_son: str,
    start_sample: int,
    end_sample: int,
    start_test: int,
    end_test: int,
    n_test,
    min_trials,
    min_neu=False,
    nonmatch=True,
    avgwin=50,
    n_sp_sec=5,
    norm=False,
    zscore=False,
    include_nid=None,
    cerotr=False,
    percentile=False,
):
    if include_nid is not None:
        nid = neu.get_neuron_id()
        if not (nid in include_nid):
            return None

    idx_start_sample = int((getattr(neu, time_before_son) + start_sample))
    idx_end_sample = int((getattr(neu, time_before_son) + end_sample))
    idx_start_test = int((getattr(neu, time_before_t1on) + start_test))
    idx_end_test = int((getattr(neu, time_before_t1on) + end_test))
    sampleon = getattr(neu, sp_son)
    t1on = getattr(neu, sp_t1on)
    mask_son = getattr(neu, mask_son)

    # Select trials aligned to sample onset

    sample_id = neu.sample_id[mask_son]

    # Build masks to select trials with match in the n_test
    mask_match = np.where(
        neu.test_stimuli[mask_son, n_test - 1] == neu.sample_id[mask_son],
        True,
        False,
    )
    mask_neu = neu.sample_id[mask_son] == 0
    # Build masks to select trials with the selected number of test presentations
    max_test = neu.test_stimuli[mask_son].shape[1]
    mask_ntest = (max_test - np.sum(np.isnan(neu.test_stimuli[mask_son]), axis=1)) > (
        n_test - 1
    )

    if nonmatch:  # include neutral and trials without a match before n_ntest (included)
        mask_match_neu = np.logical_or(mask_ntest, mask_neu)
    else:  # include neutral and trials with a match in the n_test
        mask_match_neu = np.logical_or(mask_match, mask_neu)
    if np.sum(mask_match_neu) < 20:
        return None

    # Average fr across time
    avg_sample_on = firing_rate.moving_average(
        sampleon[mask_match_neu], win=avgwin, step=1
    )[:, idx_start_sample:idx_end_sample]
    avg_test1_on = firing_rate.moving_average(t1on[mask_match_neu], win=avgwin, step=1)[
        :, idx_start_test:idx_end_test
    ]
    # Concatenate sample and test aligned data
    sp = np.concatenate((avg_sample_on, avg_test1_on), axis=1)
    # Check fr
    ms_fr = np.nanmean(sp) * 1000 > n_sp_sec
    if not ms_fr:
        return None
    # Check number of trials
    sample_id = neu.sample_id[mask_son][mask_match_neu]
    samples = [0, 11, 15, 55, 51]
    if min_neu:
        sample_fr = sp[np.where(sample_id == 0, True, False)]
        if sample_fr.shape[0] < min_trials:
            return None
    else:
        for s_id in samples:
            sample_fr = sp[np.where(sample_id == s_id, True, False)]
            if sample_fr.shape[0] < min_trials:
                return None
    if norm == True:
        sp = sp / np.max(sp)
    if zscore == True:
        sp_std = np.std(sp, ddof=1, axis=0)
        sp_std = np.where(sp_std == 0, 1, sp_std)
        sp = (sp - np.mean(sp, axis=0).reshape(1, -1)) / sp_std.reshape(1, -1)
    # Get trials grouped by sample
    fr_samples = select_trials.get_sp_by_sample(sp, sample_id, samples=samples)

    # check trials
    for isamp in fr_samples.keys():
        if ~np.all((np.isnan(fr_samples[isamp]))):
            masktr = check_trials(fr_samples[isamp], cerotr, percentile)
            fr_samples[isamp] = fr_samples[isamp][masktr]
            if fr_samples[isamp].shape[0] < min_trials:
                return None

    if fr_samples is None:
        return None

    return fr_samples
