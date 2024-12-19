from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials import select_trials
from ephysvibe.trials.spikes import firing_rate, sp_constants
from ephysvibe.spike_sorting import config
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os


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


def prepare_data_plotb1(
    neu,
    rf_stim_loc: list = ["contra", "ipsi"],
    percentile: bool = False,
    cerotr: bool = False,
):
    samples = [0, 11, 15, 55, 51]
    # IN
    sp_sampleon_0, mask_sampleon_0 = neu.align_on(
        select_block=1,
        event="sample_on",
        time_before=500,
        error_type=0,
        rf_stim_loc=rf_stim_loc[0],
    )
    samples_sampleon_0 = select_trials.get_sp_by_sample(
        sp_sampleon_0, neu.sample_id[mask_sampleon_0], samples=samples
    )
    sp_test_0, mask_test_0 = neu.align_on(
        select_block=1,
        event="test_on_1",
        time_before=500,
        error_type=0,
        rf_stim_loc=rf_stim_loc[0],
    )
    samples_test_0 = select_trials.get_sp_by_sample(
        sp_test_0, neu.sample_id[mask_test_0], samples=samples
    )
    # OUT
    sp_sampleon_1, mask_sampleon_1 = neu.align_on(
        select_block=1,
        event="sample_on",
        time_before=500,
        error_type=0,
        rf_stim_loc=rf_stim_loc[1],
    )
    samples_sampleon_1 = select_trials.get_sp_by_sample(
        sp_sampleon_1, neu.sample_id[mask_sampleon_1], samples=samples
    )
    sp_test_1, mask_test_1 = neu.align_on(
        select_block=1,
        event="test_on_1",
        time_before=500,
        error_type=0,
        rf_stim_loc=rf_stim_loc[1],
    )
    samples_test_1 = select_trials.get_sp_by_sample(
        sp_test_1, neu.sample_id[mask_test_1], samples=samples
    )

    # Check trials
    samplesstr = ["0", "11", "15", "55", "51"]
    if percentile or cerotr:
        for isamp in samplesstr:
            if ~np.all((np.isnan(samples_sampleon_0[isamp]))):
                temp = np.concatenate(
                    (
                        samples_sampleon_0[isamp][:, 300 : 500 + 450 + 400],
                        samples_test_0[isamp][:, 100 : 500 + 500],
                    ),
                    axis=1,
                )
                masknocero = np.full(temp.shape[0], True)
                maskper = np.full(temp.shape[0], True)
                if cerotr:
                    masknocero = np.sum(temp, axis=1) != 0
                if percentile:
                    maskper = select_trials_by_percentile(temp, masknocero)
                mask = np.logical_and(masknocero, maskper)
                if np.sum(mask) < 10:
                    mask = np.full(temp.shape[0], True)
                samples_sampleon_0[isamp] = samples_sampleon_0[isamp][mask]
                samples_test_0[isamp] = samples_test_0[isamp][mask]

            if ~np.all((np.isnan(samples_sampleon_1[isamp]))):
                temp = np.concatenate(
                    (
                        samples_sampleon_1[isamp][:, 300 : 500 + 450 + 400],
                        samples_test_1[isamp][:, 100 : 500 + 500],
                    ),
                    axis=1,
                )
                masknocero = np.full(temp.shape[0], True)
                maskper = np.full(temp.shape[0], True)
                if cerotr:
                    masknocero = np.sum(temp, axis=1) != 0
                if percentile:
                    maskper = select_trials_by_percentile(temp, masknocero)
                mask = np.logical_and(masknocero, maskper)
                if np.sum(mask) < 10:
                    mask = np.full(temp.shape[0], True)
                samples_sampleon_1[isamp] = samples_sampleon_1[isamp][mask]
                samples_test_1[isamp] = samples_test_1[isamp][mask]

    # Start convolution
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(
        sp_constants.W_SIZE, sp_constants.W_STD, fs=fs_ds
    )

    # IN
    conv_0 = {}
    samples_0 = {}
    for isamp in samples_sampleon_0.keys():
        if np.all((np.isnan(samples_sampleon_0[isamp]))):
            conv_0[isamp] = np.zeros((1, 1950))
            samples_0[isamp] = np.zeros((1, 1950))
            continue
        conv_sonin = (
            np.convolve(np.mean(samples_sampleon_0[isamp], axis=0), kernel, mode="same")
            * fs_ds
        )[300 : 500 + 450 + 400]

        conv_testin = (
            np.convolve(np.mean(samples_test_0[isamp], axis=0), kernel, mode="same")
            * fs_ds
        )[100 : 500 + 500]

        conv_0[isamp] = np.concatenate((conv_sonin, conv_testin))
        samples_0[isamp] = np.concatenate(
            (
                samples_sampleon_0[isamp][:, 300 : 500 + 450 + 400],
                samples_test_0[isamp][:, 100 : 500 + 500],
            ),
            axis=1,
        )

    # OUT
    conv_1 = {}
    samples_1 = {}
    for isamp in samples_sampleon_1.keys():

        if np.all((np.isnan(samples_sampleon_1[isamp]))):
            conv_1[isamp] = np.zeros((1, 1950))
            samples_1[isamp] = np.zeros((1, 1950))
            continue
        conv_sonin = (
            np.convolve(np.mean(samples_sampleon_1[isamp], axis=0), kernel, mode="same")
            * fs_ds
        )[300 : 500 + 450 + 400]
        conv_testin = (
            np.convolve(np.mean(samples_test_1[isamp], axis=0), kernel, mode="same")
            * fs_ds
        )[100 : 500 + 500]
        conv_1[isamp] = np.concatenate((conv_sonin, conv_testin))
        samples_1[isamp] = np.concatenate(
            (
                samples_sampleon_1[isamp][:, 300 : 500 + 450 + 400],
                samples_test_1[isamp][:, 100 : 500 + 500],
            ),
            axis=1,
        )
    sp = {rf_stim_loc[0]: samples_0, rf_stim_loc[1]: samples_1}
    conv = {rf_stim_loc[0]: conv_0, rf_stim_loc[1]: conv_1}

    return sp, conv


def prepare_data_plotb2(neu):
    align_sp, alig_mask = neu.align_on(
        select_block=2, event="target_on", time_before=400, error_type=0
    )
    pos_code = neu.pos_code[alig_mask]
    sp_pos = {}
    conv_pos = {}
    conv_all, n_trials = [], []
    max_n_tr = 0
    conv_max = 0
    for code in np.unique(pos_code):
        code_mask = pos_code == code
        sp_pos[str(int(code))] = align_sp[code_mask][:, 200:1900]
        mean_fr = np.mean(align_sp[code_mask], axis=0)[:2100]
        conv_fr = firing_rate.convolve_signal(mean_fr, axis=0)[200:1900]
        conv_pos[str(int(code))] = conv_fr
        conv_all.append(np.max(conv_fr))
        n_trials.append(align_sp[code_mask].shape[0])

    if len(n_trials) > 0:
        max_n_tr = np.max(n_trials)
        conv_max = np.max(conv_all)
    return sp_pos, conv_pos, max_n_tr, conv_max


def plot_trials(
    neupath: Path,
    format: str = "png",
    percentile: bool = False,
    cerotr: bool = False,
    b: int = 1,
    output_dir: Path = "./",
):

    neu = NeuronData.from_python_hdf5(neupath)
    nid = neu.get_neuron_id()
    # nid = f"{neu.subject}_{neu.area.upper()}_{neu.date_time}_{neu.cluster_group}{int(neu.cluster_number)}"
    print(nid)
    if b == 1:
        sp, conv = prepare_data_plotb1(
            neu,
            rf_stim_loc=["contra", "ipsi"],
            percentile=percentile,
            cerotr=cerotr,
        )

        fig = neu.plot_sp_b1(sp, conv)

    elif b == 2:
        sp_pos, conv_pos, max_n_tr, conv_max = prepare_data_plotb2(neu)
        fig = neu.plot_sp_b2(
            sp_pos, conv_pos, max_n_tr, conv_max, visual_rf=True, inout=1
        )

    fig.savefig(
        f"{output_dir}/{nid}.{format}",
        format=format,
        bbox_inches="tight",
        transparent=False,
    )
    plt.close(fig)
