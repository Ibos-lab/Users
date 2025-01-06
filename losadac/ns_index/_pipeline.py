import glob
import numpy as np
from typing import Dict, List
from ephysvibe.trials import firing_rate, select_trials
from ephysvibe.stats import smetrics


def check_trials(x, cerotr, percentile):
    masknocero = np.full(x.shape[0], True)
    maskper = np.full(x.shape[0], True)
    if cerotr:
        masknocero = np.sum(x, axis=1) != 0
    if percentile:
        maskper = select_trials.select_trials_by_percentile(x, masknocero)
    mask = np.logical_and(masknocero, maskper)

    return mask


def compute_roc_neutral(sp_son, sample_id, idx_start, idx_end, cerotr, percentile):
    roc_neutral = np.nan
    fr_son = firing_rate.moving_average(data=sp_son, win=100, step=1)[
        :, idx_start:idx_end
    ]
    fr_samples = select_trials.get_sp_by_sample(
        fr_son, sample_id, samples=[11, 15, 51, 55, 0]
    )
    # check trials fr
    for isamp in fr_samples.keys():
        if ~np.all((np.isnan(fr_samples[isamp]))):
            masktr = check_trials(fr_samples[isamp], cerotr, percentile)
            fr_samples[isamp] = fr_samples[isamp][masktr]
    sample = np.concatenate(
        (fr_samples["11"], fr_samples["15"], fr_samples["51"], fr_samples["55"])
    )
    if sample.shape[0] < 10:
        return roc_neutral
    n0 = fr_samples["0"]
    if np.all(np.isnan(n0)):
        return roc_neutral
    if n0.shape[0] < 10:
        return roc_neutral
    # Check selectivity and latency
    _, neutral_score, neutral_p = smetrics.get_selectivity(
        sample, n0, win=75, scores=True, sacale=False
    )
    neutral_score = neutral_score - 0.5
    sig_mask = neutral_p < 0.05
    if np.sum(sig_mask) != 0:
        iscore = np.argmax(np.abs(neutral_score))
        roc_neutral = neutral_score[iscore]
    else:  # Not sure if not set it to zero directly or search the max in all the epoch
        # iscore = np.argmax(np.abs(neutral_score[sig_mask]))
        roc_neutral = 0
    return roc_neutral


def get_screen_pos_b1b2(pos_b1, pos_b2, poscode_b2):
    # get the screen position of sample in b1 in contralateral trials
    u_pos_b1, u_count = np.unique(pos_b1, axis=0, return_counts=True)
    imax = np.argmax(u_count)
    x_pos_b1, y_pos_b1 = u_pos_b1[imax][0][0], u_pos_b1[imax][0][1]
    # Concatenate and get unique position and code during b2 trials
    pos_and_code_b2 = np.concatenate([pos_b2[:, 0], poscode_b2.reshape(-1, 1)], axis=1)
    u_pos_b2 = np.unique(pos_and_code_b2, axis=0)
    # Find the closest screen position to b1 in b2
    diff = abs(abs(u_pos_b2[:, :2]) - abs(np.array([x_pos_b1, y_pos_b1])))
    idx = np.argmin(np.sum(diff, axis=1))
    x_pos_b2 = abs(u_pos_b2[idx, 0]) * np.sign(x_pos_b1)
    y_pos_b2 = abs(u_pos_b2[idx, 1]) * np.sign(y_pos_b1)
    idx_in = np.logical_and(u_pos_b2[:, 0] == x_pos_b2, u_pos_b2[:, 1] == y_pos_b2)
    code_in = int(u_pos_b2[idx_in][0][2])
    idx_out = np.logical_and(u_pos_b2[:, 0] == -x_pos_b2, u_pos_b2[:, 1] == -y_pos_b2)
    code_out = int(u_pos_b2[idx_out][0][2])
    return code_in, code_out


def compute_roc_space(sp_pos, st_tg, end_tg, st_bl, end_bl, cerotr, percentile):
    fr = firing_rate.moving_average(data=sp_pos, win=100, step=1)
    roc_spatial = np.nan
    if ~np.all((np.isnan(fr))):
        masktr = check_trials(fr, cerotr, percentile)
        if np.sum(masktr) < 5:
            return roc_spatial
        else:
            fr = fr[masktr]
            bl = (
                np.mean(fr[:, st_bl:end_bl], axis=1)
                .reshape(-1, 1)
                .repeat(fr.shape[1], axis=1)
            )

            _, spatial_score, spatial_p = smetrics.get_selectivity(
                fr[:, st_tg:end_tg], bl, win=75, scores=True, sacale=False
            )
            spatial_score = spatial_score - 0.5
            sig_mask = spatial_p < 0.05
            if np.sum(sig_mask) != 0:
                iscore = np.argmax(np.abs(spatial_score))
                roc_spatial = spatial_score[iscore]
            else:  # Not sure if not set it to zero directly or search the max in all the epoch
                # iscore = np.argmax(np.abs(spatial_score[sig_mask]))
                roc_spatial = 0
    return roc_spatial


def get_space_neutral_roc(
    neu,
    start_sample,
    end_sample,
    st_target,
    end_target,
    st_bl,
    end_bl,
    cerotr=True,
    percentile=True,
):
    res = {}
    nid = neu.get_neuron_id()
    res["nid"] = nid
    # Neutral roc
    for inout in ["in", "out"]:
        mask = getattr(neu, "mask_son_" + inout)
        sp_son = getattr(neu, "sp_son_" + inout)
        time_before_son = getattr(neu, "time_before_son_" + inout)

        idx_start = time_before_son + start_sample
        idx_end = time_before_son + end_sample
        sample_id = neu.sample_id[mask]
        roc_neutral = compute_roc_neutral(
            sp_son, sample_id, idx_start, idx_end, cerotr, percentile
        )

        res["neutral_" + inout] = roc_neutral
    # Space roc
    b1in_mask = getattr(neu, "mask_son_in")
    b2_mask = getattr(neu, "mask_ton_")
    sp = getattr(neu, "sp_ton_")

    time_before_ton = getattr(neu, "time_before_ton_")

    ist_tg = time_before_ton + st_target
    iend_tg = time_before_ton + end_target
    ist_bl = time_before_ton + st_bl
    iend_bl = time_before_ton + end_bl
    # check b2
    if np.sum(b2_mask) == 0:
        print("no block 2 trials")  # return results
        return res
    pos_b1 = neu.position[b1in_mask]
    pos_b2 = neu.position[b2_mask]
    poscode_b2 = neu.pos_code[b2_mask]
    code_in, code_out = get_screen_pos_b1b2(pos_b1, pos_b2, poscode_b2)
    if ~np.all(neu.pos_code[b1in_mask] == neu.rf_loc[b1in_mask]):
        aux = code_in
        code_in = code_out
        code_out = aux
    for in_out, code in zip(["in", "out"], [code_in, code_out]):
        code_mask = poscode_b2 == code
        sp_pos = sp[code_mask]
        roc_spatial = compute_roc_space(
            sp_pos, ist_tg, iend_tg, ist_bl, iend_bl, cerotr, percentile
        )
        res["space_" + in_out] = roc_spatial
    return res
