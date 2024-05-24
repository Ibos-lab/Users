from ephysvibe.structures.neuron_data import NeuronData
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import glob
from ephysvibe.trials import align_trials, select_trials
from ephysvibe.trials.spikes import firing_rate
import platform
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
from pathlib import Path
from typing import Dict, List
import pca_tools
from scipy import stats
from ephysvibe.task import task_constants
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.filters import gaussian_filter1d
import scipy as sp
from matplotlib import animation
from IPython.display import HTML
import matplotlib

matplotlib.rcParams["animation.embed_limit"] = 2**128

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


def from_python_hdf5(load_path: Path):
    """Load data from a file in hdf5 format from Python."""
    with h5py.File(load_path, "r") as f:
        data = []
        for i_g in f.keys():
            group = f[i_g]
            dataset = {}
            for key, value in zip(group.keys(), group.values()):
                dataset[key] = np.array(value)
            data.append(dataset)
    f.close()
    return data


def z_score(X, with_std=False):
    # X: ndarray, shape (n_features, n_samples)
    # maxx = np.max(X,axis=1).reshape(-1,1)
    # X=X/ maxx
    ss = StandardScaler(with_mean=True, with_std=with_std)
    Xz = ss.fit_transform(X.T).T
    return Xz, ss  # ,maxx


def get_fr_samples(sp, sample_id, start, end, samples, min_trials):
    # parameters

    # Check fr
    ms_fr = np.nanmean(sp[:, start:end]) * 1000 > 5
    if not ms_fr:
        return None
    # Average spikes
    # avg_sample = firing_rate.moving_average(sp, win=win, step = 1)
    fr_samples = []
    for s_id in samples:
        sample_fr = sp[np.where(sample_id == s_id, True, False), start:end]
        # Check number of trials
        if sample_fr.shape[0] < min_trials:
            return None
        fr_samples.append(np.mean(sample_fr, axis=0))
    return fr_samples


def get_neuron_sample_fr(path, time_before, start, end, min_trials):
    neu_data = NeuronData.from_python_hdf5(path)
    select_block = 1
    code = 1
    idx_start = time_before + start
    idx_end = time_before + end
    # Select trials aligned to sample onset
    sp_sample_on, mask = align_trials.align_on(
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
        error_type=0,
    )
    if np.sum(mask) < 20:
        return {"fr": None}
    sample_id = neu_data.sample_id[mask]
    fr_samples = get_fr_samples(
        sp_sample_on,
        sample_id,
        start=idx_start,
        end=idx_end,
        samples=[0, 11, 15, 55, 51],
        min_trials=min_trials,
    )
    if fr_samples is None:
        return {"fr": None}
    return {"fr": fr_samples}


def select_alltrials(neu_data, select_block, code, time_before, error_type=0):
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
        time_before=time_before + 400,
        error_type=error_type,
    )
    if np.any(mask_sample != mask_test1):
        return "error"
    return sp_sample_on, sp_test1_on, mask_sample, mask_test1


def get_neuron_sample_test1_fr(
    path, time_before, start, end, end_test, n_test, min_trials, nonmatch=True, win=50
):
    neu_data = NeuronData.from_python_hdf5(path)
    select_block = 1
    code = 1
    idx_start = time_before + start
    idx_end = time_before + end
    # Select trials aligned to sample onset
    sp_sample_on, sp_test1_on, mask_sample, mask_test1 = select_alltrials(
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

    if nonmatch:
        mask_match_neu = np.logical_or(mask_ntest, mask_neu)
    else:
        mask_match_neu = np.logical_or(mask_match, mask_neu)
    if np.sum(mask_match_neu) < 20:
        return {"fr": None}

    avg_sample_on = firing_rate.moving_average(
        sp_sample_on[mask_match_neu], win=win, step=1
    )[:, : time_before + 450 + 400]
    avg_test1_on = firing_rate.moving_average(
        sp_test1_on[mask_match_neu], win=win, step=1
    )[:, time_before : time_before + end_test + 400]
    sp = np.concatenate((avg_sample_on, avg_test1_on), axis=1)[:, idx_start:idx_end]
    # Check fr
    ms_fr = np.nanmean(sp) * 1000 > 1
    if not ms_fr:
        return {"fr": None}

    sample_id = neu_data.sample_id[mask_test1][mask_match_neu]
    samples = [0, 11, 15, 55, 51]
    for s_id in samples:
        sample_fr = sp[np.where(sample_id == s_id, True, False)]
        # Check number of trials
        if sample_fr.shape[0] < min_trials:
            return {"fr": None}

    fr_samples = select_trials.get_sp_by_sample(sp, sample_id, samples=samples)

    if fr_samples is None:
        return {"fr": None}
    return {"fr": fr_samples}


greyscale = [
    "#000000",
    "#0F0F0F",
    "#3D3D3D",
    "#6B6B6B",
    "#7A7A7A",
    "#999999",
    "#C7C7C7",
    "#D6D6D6",
]
colors_g = [
    "#001755",
    "#002B95",
    "#0036B3",
    "#1A54D8",
    "#3566DF",
    "#4F79E5",
    "#84A0F0",
    "#9EB4F5",
]

# Load data
n_test = 1
min_trials = 10
nonmatch = True
time_before = 500
start = -200
end_test = n_test * 450 + 200
end = 450 + 400 + 400 + end_test
win = 200
# Define epochs
part1 = 200 + 450 + 400
test1_st = part1 + 400
test2_st = test1_st + 450
test3_st = test2_st + 450
test4_st = test3_st + 450
test5_st = test4_st + 450
idx_f = np.arange(0, 200, 2)
idx_s = np.arange(200, 200 + 450, 2)
idx_d1 = np.arange(200 + 450, part1, 2)
idx_d2 = np.arange(part1, test1_st, 2)
idx_t1 = np.arange(test1_st, test2_st, 2)
idx_t2 = np.arange(test2_st, test3_st, 2)
idx_t3 = np.arange(test3_st, test4_st, 2)
idx_t4 = np.arange(test4_st, test5_st, 2)
idx_aftert = np.arange(test2_st, end - start, 2)

area = "lip"
subject = "Riesling"
save = True
savepath = "./"
save_format = "png"

# part 1
t_epochs1 = {
    "fixation": idx_f,
    "sample": idx_s,
    "delay": idx_d1,
}  # ,'test3':idx_t3,'test4':idx_t4 'test2':idx_t2,
colors1 = {
    "epochs": {"fixation": colors_g[0], "sample": colors_g[1], "delay": colors_g[1]}
}
colors_neu1 = {
    "epochs": {"fixation": greyscale[0], "sample": greyscale[1], "delay": greyscale[1]}
}  #
# part 2
t_epochs2 = {
    "delay": idx_d2,
    "test1": idx_t1,
    "test off": idx_aftert,
}  # ,'test3':idx_t3,'test4':idx_t4 'test2':idx_t2,
colors2 = {
    "epochs": {
        "delay": colors_g[6],
        "test1": colors_g[6],
        "test2": colors_g[6],
        "test3": colors_g[6],
        "test4": colors_g[6],
        "test off": colors_g[-1],
    }
}
colors_neu2 = {
    "epochs": {
        "delay": greyscale[5],
        "test1": greyscale[5],
        "test2": greyscale[5],
        "test3": greyscale[5],
        "test4": greyscale[5],
        "test off": greyscale[7],
    }
}  #

if platform.system() == "Linux":
    basepath = (
        "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/"
    )
elif platform.system() == "Windows":
    basepath = "C:/Users/camil/Documents/int/"

neu_path = basepath + "/session_struct/" + area + "/neurons/*neu.h5"
all_path_list = glob.glob(neu_path)

all_sessions = []
for ipath in all_path_list:
    all_sessions.append(ipath.replace("\\", "/").split("/")[-1][:19])
all_sessions = np.unique(all_sessions)

for session in all_sessions:
    session = str(session)
    print(session)
    mask = []
    for path in all_path_list:
        mask.append(
            session == path.replace("\\", "/").rsplit("/")[-1].rsplit("_" + subject)[0]
        )
    path_list = np.array(all_path_list)[np.array(mask)]

    data = Parallel(n_jobs=-1)(
        delayed(get_neuron_sample_test1_fr)(
            path, time_before, start, end, end_test, n_test, min_trials, nonmatch, win
        )
        for path in tqdm(path_list)
    )

    if nonmatch:
        pname_match = "_wnonmatch_"
    else:
        pname_match = "_match_"

    s0, s11, s15, s51, s55 = [], [], [], [], []
    s0mean, s11mean, s15mean, s51mean, s55mean = [], [], [], [], []
    for asc in data:
        fr = asc["fr"]
        if fr is not None:
            s0.append(fr["0"])
            s11.append(fr["11"])
            s15.append(fr["15"])
            s51.append(fr["51"])
            s55.append(fr["55"])
            s0mean.append(np.mean(fr["0"], axis=0))
            s11mean.append(np.mean(fr["11"], axis=0))
            s15mean.append(np.mean(fr["15"], axis=0))
            s51mean.append(np.mean(fr["51"], axis=0))
            s55mean.append(np.mean(fr["55"], axis=0))

    neurons_fr = [
        {
            "0mean": s0mean,
            "11mean": s11mean,
            "15mean": s15mean,
            "51mean": s51mean,
            "55mean": s55mean,
            "0": s0,
            "11": s11,
            "15": s15,
            "51": s51,
            "55": s55,
        }
    ]

    if len(s55) < 6:
        continue
    basepath = "./"  #'/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/pca/data/'+area +'/'
    to_python_hdf5(
        dat=neurons_fr,
        save_path=basepath
        + area
        + "_"
        + session
        + "_trials_win"
        + str(win)
        + "_test"
        + str(n_test)
        + pname_match
        + "min"
        + str(min_trials)
        + "tr_pca.h5",
    )

    neurons_fr_read = from_python_hdf5(
        load_path=basepath
        + area
        + "_"
        + session
        + "_trials_win"
        + str(win)
        + "_test"
        + str(n_test)
        + pname_match
        + "min"
        + str(min_trials)
        + "tr_pca.h5"
    )
    neurons_fr = neurons_fr_read[0]

    n_comp = len(neurons_fr["11"])
    all_mean = np.concatenate(
        (
            neurons_fr["0mean"],
            neurons_fr["11mean"],
            neurons_fr["15mean"],
            neurons_fr["51mean"],
            neurons_fr["55mean"],
        ),
        axis=1,
    )
    all_mean, ss = z_score(all_mean, with_std=False)
    model, pc_s = pca_tools.compute_pca(all_mean, n_comp=n_comp)

    fig, ax = plt.subplots(figsize=(5, 5))
    pca_tools.plot_explained_var(model, figsize=(5, 5), area=area, fig=fig, ax=ax)
    if save:
        figname = area + session + "explained_var." + save_format
        fig.savefig(savepath + figname, format=save_format, bbox_inches="tight")
    reshape_pc_s = pc_s.reshape(n_comp, -1, end - start)
    mean_pc_s = np.mean(reshape_pc_s[:, 1:, :], axis=1)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    pca_tools.plot_pc_neu(
        mean_pc_s,
        reshape_pc_s[:, 0],
        colors1,
        colors_neu1,
        t_epochs1,
        area,
        fig=fig,
        ax=ax,
        idot=0,
    )
    pca_tools.plot_pc_neu(
        mean_pc_s,
        reshape_pc_s[:, 0],
        colors2,
        colors_neu2,
        t_epochs2,
        area,
        fig=fig,
        ax=ax,
        idot=-1,
    )

    if save:
        figname = area + session + "_pcs_neutral_sample." + save_format
        fig.savefig(savepath + figname, format=save_format, bbox_inches="tight")

    mean_pc_s = np.mean(reshape_pc_s[:, 1:, :], axis=1)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    pca_tools.plot_pc(
        mean_pc_s, colors1, t_epochs1, area, sample_flag=False, fig=fig, ax=ax, idot=0
    )
    pca_tools.plot_pc(
        mean_pc_s, colors2, t_epochs2, area, sample_flag=False, fig=fig, ax=ax, idot=-1
    )
    if save:
        figname = area + session + "_pcs_sample_mean." + save_format
        fig.savefig(savepath + figname, format=save_format, bbox_inches="tight")

    ## Transform trials
    ssneurons_fr = {}
    for i_s in ("0", "11", "15", "51", "55"):
        emptyarr = np.zeros(neurons_fr[i_s].shape)
        for itr in range(neurons_fr[i_s].shape[1]):
            emptyarr[:, itr, :] = ss.transform((neurons_fr[i_s][:, itr]).T).T
        ssneurons_fr[i_s] = emptyarr

    proj_sam = {}
    for i_s in ("0", "11", "15", "51", "55"):

        tr_proj = model.transform(ssneurons_fr[i_s].reshape(n_comp, -1).T).T
        tr_proj = tr_proj.reshape(n_comp, -1, ssneurons_fr[i_s].shape[-1])
        proj_sam[i_s] = tr_proj

    color1 = [
        "#0A4580",
        "#0F8792",
        "#13A272",
        "#18B13C",
        "#3EBF1E",
        "#8ECD25",
        "#DAD02C",
        "#E69034",
        "#EC3E3B",
        "#F14297",
        "#F54AF2",
        "#A953F9",
        "#5E5CFD",
        "#65B0FF",
    ]
    color2 = [
        "#81A2C3",
        "#84C6CC",
        "#85D4BA",
        "#87DC9B",
        "#9CE38A",
        "#C8EB8E",
        "#F1EC91",
        "#F7C895",
        "#FA9A98",
        "#F14297",
        "#FEB4D8",
        "#DFBAFF",
        "#BFBEFF",
        "#C1E1FF",
    ]
    sigma = 3
    darkcolor = {
        "0": "k",
        "11": "firebrick",  # o1_c1
        "15": "teal",  # o1_c5
        "51": "tomato",  # o5_c1
        "55": "c",  # "lightseagreen",  #
    }

    part1 = 200 + 450 + 400
    fig, ax = plt.subplots(3, 2, figsize=(18, 10))
    for i_sample in ["0", "11", "15", "51", "55"]:
        trialspc = proj_sam[i_sample]

        for itr in range(0, trialspc.shape[1], 10):
            pcs = trialspc[:, itr]
            irow, icol = 0, 0
            for i_c in range(6):
                if irow == 3:
                    irow, icol = 0, 1
                ax[irow, icol].plot(
                    range(start, end)[:part1],
                    pcs[i_c, :part1],
                    color=task_constants.PALETTE_B1[i_sample],
                    label=i_sample,
                    alpha=0.1,
                )
                ax[irow, icol].plot(
                    range(start, end)[part1:],
                    pcs[i_c, part1:],
                    color=task_constants.PALETTE_B1[i_sample],
                    label=i_sample,
                    alpha=0.1,
                )
                ax[irow, icol].set(xlabel="Time(ms)", ylabel="PC" + str(i_c + 1))
                irow = irow + 1
    for i_sample in ["0", "11", "15", "51", "55"]:
        trialspc = proj_sam[i_sample]
        irow, icol = 0, 0
        for i_c in range(6):
            if irow == 3:
                irow, icol = 0, 1
            ax[irow, icol].plot(
                range(start, end)[:part1],
                np.mean(trialspc[i_c, :, :part1], axis=0),
                color=darkcolor[i_sample],
                label=i_sample,
                alpha=1,
            )
            ax[irow, icol].plot(
                range(start, end)[part1:],
                np.mean(trialspc[i_c, :, part1:], axis=0),
                color=darkcolor[i_sample],
                label=i_sample,
                alpha=1,
            )
            irow = irow + 1
    if save:
        figname = area + session + "_pcs_time_trial." + save_format
        fig.savefig(savepath + figname, format=save_format, bbox_inches="tight")

    part1 = 200 + 450 + 400
    fig, ax = plt.subplots(3, 2, figsize=(18, 10))
    for i_sample in ["0", "11", "15", "51", "55"]:
        trialspc = proj_sam[i_sample]
        irow, icol = 0, 0
        for i_c in range(6):
            if irow == 3:
                irow, icol = 0, 1
            h = np.std(trialspc[i_c, :, :part1], axis=0)
            ax[irow, icol].plot(
                range(start, end)[:part1],
                np.mean(trialspc[i_c, :, :part1], axis=0),
                color=darkcolor[i_sample],
                label=i_sample,
                alpha=1,
            )
            ax[irow, icol].fill_between(
                range(start, end)[:part1],
                np.mean(trialspc[i_c, :, :part1], axis=0) + h,
                np.mean(trialspc[i_c, :, :part1], axis=0) - h,
                color=task_constants.PALETTE_B1[i_sample],
                alpha=0.1,
            )
            ## part 2
            h = np.std(trialspc[i_c, :, part1:], axis=0)
            ax[irow, icol].plot(
                range(start, end)[part1:],
                np.mean(trialspc[i_c, :, part1:], axis=0),
                color=darkcolor[i_sample],
                label=i_sample,
                alpha=1,
            )
            ax[irow, icol].fill_between(
                range(start, end)[part1:],
                np.mean(trialspc[i_c, :, part1:], axis=0) + h,
                np.mean(trialspc[i_c, :, part1:], axis=0) - h,
                color=task_constants.PALETTE_B1[i_sample],
                alpha=0.1,
            )
            ax[irow, icol].set(xlabel="Time(ms)", ylabel="PC" + str(i_c + 1))
            irow = irow + 1
    if save:
        figname = area + session + "_pcs_time_std." + save_format
        fig.savefig(savepath + figname, format=save_format, bbox_inches="tight")

    # Animation
    color_epo = ["#0A4580", "#18B13C", "#DAD02C", "#F14297", "#5E5CFD", "#65B0FF"]

    for i_sample in ["0", "11", "15", "51", "55"]:
        trialspc = proj_sam[i_sample]
        # apply some smoothing to the trajectories
        sigma = 3  # smoothing amount

        # mean_pc_s= np.mean(reshape_pc_s[:,1:,:],axis=1)
        g_mean_pc = trialspc.copy()

        for c in range(g_mean_pc.shape[0]):
            for i_tr in range(g_mean_pc.shape[1]):
                g_mean_pc[c, i_tr, :] = gaussian_filter1d(
                    g_mean_pc[c, i_tr, :], sigma=sigma
                )

        # g_mean_pc=g_mean_pc.reshape(n_comp,-1,end-start)

        x1 = g_mean_pc[:, :, :-200].copy()
        x1[:, :, 200 + 450 + 400 :] = np.nan
        x2 = g_mean_pc[:, :, 200:].copy()
        x2[:, :, : 450 + 400] = np.nan

        time = np.arange(0, x2.shape[-1]) - 200

        # utility function to clean up and label the axes
        def style_ax(ax, xlabel, ylabel, i_sample):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title("%s - sample %s" % (area.upper(), i_sample))

        mean_pc_s = np.mean(reshape_pc_s[:, 1:, :], axis=1)
        for c in range(mean_pc_s.shape[0]):
            mean_pc_s[c, :] = gaussian_filter1d(mean_pc_s[c, :], sigma=sigma)

        for component_x in np.arange(0, 2):
            for component_y in np.arange(0 + component_x, 2):
                if component_x == component_y:
                    continue
                # create the figure
                fig = plt.figure(figsize=[9, 9])
                ax = fig.add_subplot()  # 1, 1, projection='3d')
                plt.close()

                # annotate with stimulus and time information
                text = ax.text(
                    0.05,
                    0.95,
                    "Fixation \nt = %d ms" % (time[0]),
                    fontsize=14,
                    transform=ax.transAxes,
                    va="top",
                )

                def animate(i):
                    # pick the components corresponding to the x, y, and z axes
                    # component_x = 1
                    # component_y = 2
                    ax.clear()  # clear up trajectories from previous iteration
                    xlabel = "PC " + str(component_x + 1)
                    ylabel = "PC " + str(component_y + 1)
                    style_ax(ax, xlabel, ylabel, i_sample)
                    j = 0
                    if i > 200:
                        j = i - 200

                    # plot avg
                    color = task_constants.PALETTE_B1[i_sample]
                    ax.plot(
                        mean_pc_s[component_x, : 450 + 400],
                        mean_pc_s[component_y, : 450 + 400],
                        color,
                    )
                    ax.plot(
                        mean_pc_s[component_x, 200 + 450 + 400 :],
                        mean_pc_s[component_y, 200 + 450 + 400 :],
                        color,
                        linestyle="--",
                    )

                    for ic, itr in enumerate(range(0, g_mean_pc.shape[1], 10)):

                        # plot 1st part
                        x11 = x1[component_x, itr, 0:i]
                        y11 = x1[component_y, itr, 0:i]
                        ax.plot(x11[j:i], y11[j:i], color1[ic])
                        # plot 2nd part
                        x22 = x2[component_x, itr, 0:i]
                        y22 = x2[component_y, itr, 0:i]
                        ax.plot(x22[j:i], y22[j:i], color2[ic])

                        xlim = [x1[component_x], x2[component_x]]
                        ylim = [x1[component_y], x2[component_y]]
                        ax.set_xlim((np.nanmin(xlim) - 0.005, np.nanmax(xlim) + 0.005))
                        ax.set_ylim((np.nanmin(ylim) - 0.005, np.nanmax(ylim) + 0.005))
                        # update stimulus and time annotation
                        if i < 200:
                            # stimdot.set_data(10, 14)
                            if itr == 0:
                                ax.text(
                                    0.05,
                                    0.95,
                                    "Fixation \nt = %d ms" % (time[i]),
                                    fontsize=14,
                                    transform=ax.transAxes,
                                    va="top",
                                )
                        elif np.logical_and(i > 200, i < 200 + 450):
                            # stimdot.set_data(10, 14)
                            ax.scatter(x11[200], y11[200], c=color_epo[0])
                            if itr == 0:
                                ax.text(
                                    0.05,
                                    0.95,
                                    "Sample ON \nt = %d ms" % (time[i]),
                                    fontsize=14,
                                    transform=ax.transAxes,
                                    va="top",
                                )
                        elif np.logical_and(i > 200 + 450, i < 200 + 450 + 600):
                            # stimdot.set_data(10, 14)
                            ax.scatter(x11[200], y11[200], c=color_epo[0])
                            ax.scatter(x11[200 + 450], y11[200 + 450], c=color_epo[1])
                            if itr == 0:
                                ax.text(
                                    0.05,
                                    0.95,
                                    "Delay \nt = %d ms" % (time[i]),
                                    fontsize=14,
                                    transform=ax.transAxes,
                                    va="top",
                                )
                        elif np.logical_and(
                            i > 200 + 450 + 600, i < 200 + 450 + 600 + 450
                        ):
                            # stimdot.set_data(10, 14)
                            ax.scatter(x11[200], y11[200], c=color_epo[0])
                            ax.scatter(x11[200 + 450], y11[200 + 450], c=color_epo[1])
                            ax.scatter(
                                x22[200 + 450 + 600],
                                y22[200 + 450 + 600],
                                c=color_epo[2],
                            )
                            if itr == 0:
                                ax.text(
                                    0.05,
                                    0.95,
                                    "Test ON \nt = %d ms" % (time[i]),
                                    fontsize=14,
                                    transform=ax.transAxes,
                                    va="top",
                                )
                        else:
                            # stimdot.set_data([], [])
                            ax.scatter(x11[200], y11[200], c=color_epo[0])
                            ax.scatter(x11[200 + 450], y11[200 + 450], c=color_epo[1])
                            ax.scatter(
                                x22[200 + 450 + 600],
                                y22[200 + 450 + 600],
                                c=color_epo[2],
                            )
                            ax.scatter(
                                x22[200 + 450 + 600 + 450],
                                y22[200 + 450 + 600 + 450],
                                c=color_epo[3],
                            )
                            if itr == 0:
                                ax.text(
                                    0.05,
                                    0.95,
                                    "Test OFF \nt = %d ms" % (time[i]),
                                    fontsize=14,
                                    transform=ax.transAxes,
                                    va="top",
                                )
                    return []

                anim = animation.FuncAnimation(
                    fig,
                    animate,
                    frames=range(1, x1.shape[-1], 2),
                    interval=50,
                    blit=True,
                )

                #
                if save:
                    figname = (
                        area
                        + "_"
                        + i_sample
                        + "_pc"
                        + str(component_x + 1)
                        + "_pc"
                        + str(component_y + 1)
                        + "_trials_animation.gif"
                    )
                    anim.save(figname, writer="Pillow", fps=30)
                else:
                    HTML(anim.to_jshtml(default_mode="once"))
