from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials import select_trials
from ephysvibe.trials.spikes import firing_rate, sp_constants
from ephysvibe.spike_sorting import config
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def select_trials_by_percentile(x: np.ndarray):
    ntr = x.shape[0]
    if ntr < 2:
        return np.full(ntr, True)
    mean_trs = np.mean(x, axis=1)

    q25, q75 = np.percentile(mean_trs, [25, 75])
    iqr = q75 - q25
    upper_limit = q75 + 1.5 * iqr
    lower_limit = q25 - 1.5 * iqr

    q1mask = mean_trs > lower_limit
    q2mask = mean_trs < upper_limit

    mask = np.logical_and(q1mask, q2mask)
    return mask


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
    if percentile or cerotr:
        for sample in samples_sampleon_0.keys():
            if ~np.all((np.isnan(samples_sampleon_0[sample]))):
                temp = np.concatenate(
                    (
                        samples_sampleon_0[sample][:, 300 : 500 + 450 + 400],
                        samples_test_0[sample][:, 100 : 500 + 500],
                    ),
                    axis=1,
                )
                masknocero = np.full(temp.shape[0], True)
                maskper = np.full(temp.shape[0], True)
                if cerotr:
                    masknocero = np.sum(temp, axis=1) != 0
                if percentile:
                    maskper = select_trials_by_percentile(temp)
                mask = np.logical_and(masknocero, maskper)
                if np.sum(mask) < 10:
                    mask = np.full(temp.shape[0], True)
                samples_sampleon_0[sample] = samples_sampleon_0[sample][mask]
                samples_test_0[sample] = samples_test_0[sample][mask]

                if np.all(np.isnan(samples_sampleon_1[sample])):
                    samples_sampleon_1[sample] = np.zeros((2, 1950))
                if np.all(np.isnan(samples_test_1[sample])):
                    samples_test_1[sample] = np.zeros((2, 1950))

            if ~np.all((np.isnan(samples_sampleon_1[sample]))):
                temp = np.concatenate(
                    (
                        samples_sampleon_1[sample][:, 300 : 500 + 450 + 400],
                        samples_test_1[sample][:, 100 : 500 + 500],
                    ),
                    axis=1,
                )
                masknocero = np.full(temp.shape[0], True)
                maskper = np.full(temp.shape[0], True)
                if cerotr:
                    masknocero = np.sum(temp, axis=1) != 0
                if percentile:
                    maskper = select_trials_by_percentile(temp)
                mask = np.logical_and(masknocero, maskper)
                if np.sum(mask) < 10:
                    mask = np.full(temp.shape[0], True)
                samples_sampleon_1[sample] = samples_sampleon_1[sample][mask]
                samples_test_1[sample] = samples_test_1[sample][mask]

    # Start convolution
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(
        sp_constants.W_SIZE, sp_constants.W_STD, fs=fs_ds
    )

    # IN
    conv_0 = {}
    samples_0 = {}
    for sample in samples_sampleon_0.keys():
        if np.all((np.isnan(samples_sampleon_0[sample]))):
            conv_0[sample] = np.zeros((1, 1950))
            samples_0[sample] = np.zeros((1, 1950))
            continue
        conv_sonin = (
            np.convolve(
                np.mean(samples_sampleon_0[sample], axis=0), kernel, mode="same"
            )
            * fs_ds
        )[300 : 500 + 450 + 400]

        conv_testin = (
            np.convolve(np.mean(samples_test_0[sample], axis=0), kernel, mode="same")
            * fs_ds
        )[100 : 500 + 500]

        conv_0[sample] = np.concatenate((conv_sonin, conv_testin))
        samples_0[sample] = np.concatenate(
            (
                samples_sampleon_0[sample][:, 300 : 500 + 450 + 400],
                samples_test_0[sample][:, 100 : 500 + 500],
            ),
            axis=1,
        )

    # OUT
    conv_1 = {}
    samples_1 = {}
    for sample in samples_sampleon_1.keys():

        if np.all((np.isnan(samples_sampleon_1[sample]))):
            conv_0[sample] = np.zeros((1, 1950))
            samples_0[sample] = np.zeros((1, 1950))
            continue
        conv_sonin = (
            np.convolve(
                np.mean(samples_sampleon_1[sample], axis=0), kernel, mode="same"
            )
            * fs_ds
        )[300 : 500 + 450 + 400]
        conv_testin = (
            np.convolve(np.mean(samples_test_1[sample], axis=0), kernel, mode="same")
            * fs_ds
        )[100 : 500 + 500]
        conv_1[sample] = np.concatenate((conv_sonin, conv_testin))
        samples_1[sample] = np.concatenate(
            (
                samples_sampleon_1[sample][:, 300 : 500 + 450 + 400],
                samples_test_1[sample][:, 100 : 500 + 500],
            ),
            axis=1,
        )
    sp = {rf_stim_loc[0]: samples_0, rf_stim_loc[1]: samples_1}
    conv = {rf_stim_loc[0]: conv_0, rf_stim_loc[1]: conv_1}

    return sp, conv


def plot_trials(
    neupath: Path, format: str = "png", percentile: bool = False, cerotr: bool = False
):

    neu = NeuronData.from_python_hdf5(neupath)
    nid = neu.get_neuron_id()
    print(nid)
    sp, conv = prepare_data_plotb1(
        neu,
        rf_stim_loc=["contra", "ipsi"],
        percentile=percentile,
        cerotr=cerotr,
    )

    fig = neu.plot_sp_b1(sp, conv)
    fig.savefig(
        f"{nid}.{format}",
        format=format,
        bbox_inches="tight",
        transparent=False,
    )
    plt.close(fig)
