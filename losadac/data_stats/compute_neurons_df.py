import numpy as np
from scipy import stats
from ephysvibe.trials import align_trials
from collections import defaultdict
from typing import Dict
from sklearn import metrics
from ephysvibe.structures.neuron_data import NeuronData
import pandas as pd
import platform
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path


## Def functions
def moving_average(data: np.ndarray, win: int, step: int = 1) -> np.ndarray:
    d_shape = data.shape
    d_avg = np.zeros((d_shape[0], int(np.floor(d_shape[1] / step))))
    count = 0
    for i_step in np.arange(0, d_shape[1] - step, step):
        d_avg[:, count] = np.mean(data[:, i_step : i_step + win], axis=1)
        count += 1
    return d_avg


# select sp by sample feature
def get_sp_by_sample(sp, samples, sample_id):
    sp_samples = {}
    for s_id in samples:
        s_sp = sp[np.where(sample_id == s_id, True, False)]
        # Check number of trials
        if s_sp.shape[0] > 0:
            sp_samples[str(s_id)] = s_sp
        else:
            sp_samples[str(s_id)] = np.array([np.nan])
    return sp_samples


def get_avg_fr(sp):
    fr = np.nanmean(sp) * 1000
    return fr


def scale_p(x, out_range=(-1, 1)):
    if np.sum(x > 1) > 0:
        return
    domain = 0, 1
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def compute_roc_auc(group1, group2):
    roc_score = []
    p = []
    for n_win in np.arange(group1.shape[1]):
        g1 = group1[:, n_win]
        g2 = group2[:, n_win]
        p.append(stats.ranksums(g1, g2)[1])
        thresholds = np.unique(np.concatenate([g1, g2]))
        y_g1, y_g2 = np.ones(len(g1)), np.zeros(len(g2))
        score = 0.5
        fpr, tpr = [], []
        for threshold in thresholds:
            g1_y_pred, g2_y_pred = np.zeros(len(g1)), np.zeros(len(g2))
            g1_mask, g2_mask = g1 >= threshold, g2 >= threshold
            g1_y_pred[g1_mask], g2_y_pred[g2_mask] = 1, 1
            tp = sum(np.logical_and(y_g1 == 1, g1_y_pred == 1))
            fn = sum(np.logical_and(y_g1 == 1, g1_y_pred == 0))
            tpr.append(tp / (tp + fn))
            fp = sum(np.logical_and(y_g2 == 0, g2_y_pred == 1))
            tn = sum(np.logical_and(y_g2 == 0, g2_y_pred == 0))
            fpr.append(fp / (fp + tn))
        if len(fpr) > 1:
            fpr, tpr = np.array(fpr), np.array(tpr)
            score = metrics.auc(fpr[fpr.argsort()], tpr[fpr.argsort()])
        roc_score.append(score)
    roc_score = np.array(roc_score)
    roc_score = scale_p(np.round(roc_score, 2), out_range=[-1, 1])
    return roc_score, np.array(p)


def find_latency(p_value: np.ndarray, win: int, step: int = 1) -> np.ndarray:
    sig = np.full(p_value.shape[0], False)
    sig[p_value < 0.05] = True
    for i_step in np.arange(0, sig.shape[0], step):
        sig[i_step] = np.where(
            np.all(p_value[i_step : i_step + win] < 0.05), True, False
        )
    latency = np.where(sig)[0]

    if len(latency) != 0:
        endl = np.where(np.cumsum(sig[latency[0] :]) == 0)[0]
        endl = endl[0] if len(endl) != 0 else -1
        return latency[0], endl
    else:
        return np.nan, np.nan


def get_selectivity(sp_1, sp_2, win):
    if np.logical_or(sp_1.ndim < 2, sp_2.ndim < 2):
        return np.nan, np.nan
    if np.logical_or(sp_1.shape[0] < 2, sp_2.shape[0] < 2):
        return np.nan, np.nan
    roc_score, p_value = compute_roc_auc(sp_1, sp_2)
    lat, _ = find_latency(p_value, win=win, step=1)
    if np.isnan(lat):
        return lat, np.nan
    return lat, roc_score[lat]


def get_align_tr(neu_data, select_block, select_pos, time_before, event="sample_on"):
    sp, mask = align_trials.align_on(
        sp_samples=neu_data.sp_samples,
        code_samples=neu_data.code_samples,
        code_numbers=neu_data.code_numbers,
        trial_error=neu_data.trial_error,
        block=neu_data.block,
        pos_code=neu_data.pos_code,
        select_block=select_block,
        select_pos=select_pos,
        event=event,
        time_before=time_before,
        error_type=0,
    )
    return sp, mask


# def get_vd_idx(bs, sp_sample, sp_delay):
#     bs_fr = get_avg_fr(bs)
#     sample_fr = get_avg_fr(sp_sample)
#     delay_fr = get_avg_fr(sp_delay)
#     sample_fr = np.abs(sample_fr - bs_fr)
#     delay_fr = np.abs(delay_fr - bs_fr)
#     return (delay_fr - sample_fr) / (delay_fr + sample_fr)


def get_vd_index(bl, group1, group2, step=1, avg_win=100, pwin=75):
    p_son, p_d = [], []
    bl = np.mean(bl, axis=1)
    for i in range(0, group1.shape[1] - avg_win, step):
        g1 = np.mean(group1[:, i : i + avg_win], axis=1)
        p_son.append(stats.ranksums(bl, g1)[1])
    for i in range(0, group2.shape[1] - avg_win, step):
        g2 = np.mean(group2[:, i : i + avg_win], axis=1)
        p_d.append(stats.ranksums(bl, g2)[1])
    p_son = np.array(p_son)
    p_d = np.array(p_d)
    lat_son, end_son = find_latency(p_value=p_son, win=pwin, step=1)
    lat_d, end_d = find_latency(p_value=p_d, win=pwin, step=1)
    if np.logical_and(np.isnan(lat_son), ~np.isnan(lat_d)):
        g1 = group1
        g2 = group2[:, lat_d:end_d]
    elif np.logical_and(~np.isnan(lat_son), np.isnan(lat_d)):
        g1 = group1[:, lat_son:end_son]
        g2 = group2
    elif np.logical_and(np.isnan(lat_son), np.isnan(lat_d)):
        return np.nan, np.nan, np.nan, np.nan
    else:
        g1 = group1[:, lat_son:end_son]
        g2 = group2[:, lat_d:end_d]
    bl_mean = np.mean(bl)
    g1_mean = np.mean(g1)
    g2_mean = np.mean(g2)
    g2_mean_bl = np.abs(g2_mean - bl_mean)
    g1_mean_bl = np.abs(g1_mean - bl_mean)
    vd_idx = (g2_mean_bl - g1_mean_bl) / (g1_mean_bl + g2_mean_bl)
    return vd_idx, bl_mean, g1_mean, g2_mean


def compute_vd_idx(neu_data):
    time_before = 200
    # get spike matrices in and out conditions
    sp_in, mask_in = get_align_tr(
        neu_data, select_block=1, select_pos=1, time_before=time_before
    )
    sp_in = sp_in[neu_data.sample_id[mask_in] != 0]
    sp_out, mask_out = get_align_tr(
        neu_data, select_block=1, select_pos=-1, time_before=time_before
    )
    sp_out = sp_out[neu_data.sample_id[mask_out] != 0]
    sp_din, mask_din = get_align_tr(
        neu_data, select_block=1, select_pos=1, time_before=0, event="sample_off"
    )
    sp_din = sp_din[neu_data.sample_id[mask_din] != 0]
    sp_dout, mask_dout = get_align_tr(
        neu_data, select_block=1, select_pos=-1, time_before=0, event="sample_off"
    )
    sp_dout = sp_dout[neu_data.sample_id[mask_dout] != 0]

    #### Compute VD index
    # get avg fr over trials and time
    vd_in, bl_in, g1_in, g2_in = np.nan, np.nan, np.nan, np.nan
    vd_out, bl_out, g1_out, g2_out = np.nan, np.nan, np.nan, np.nan
    i_st = 10
    if np.logical_and(sp_din.shape[0] > 2, sp_din.ndim > 1):
        vd_in, bl_in, g1_in, g2_in = get_vd_index(
            bl=sp_in[:, :time_before],
            group1=sp_in[:, time_before + i_st : time_before + i_st + 460],
            group2=sp_din[:, i_st:400],
            step=1,
            avg_win=200,
            pwin=150,
        )
    if np.logical_and(sp_dout.shape[0] > 2, sp_dout.ndim > 1):
        vd_out, bl_out, g1_out, g2_out = get_vd_index(
            bl=sp_out[:, :time_before],
            group1=sp_out[:, time_before + i_st : time_before + i_st + 460],
            group2=sp_dout[:, i_st:400],
            step=1,
            avg_win=200,
            pwin=150,
        )
    return vd_in, vd_out, bl_in, g1_in, g2_in, bl_out, g1_out, g2_out


def get_neuron_info(path: Path) -> Dict:
    neu_info = {}
    # Read neuron data
    neu_data = NeuronData.from_python_hdf5(path)
    time_before = 200
    win = 75
    # get spike matrices in and out conditions
    sp_in, mask_in = get_align_tr(
        neu_data, select_block=1, select_pos=1, time_before=time_before
    )
    sp_out, mask_out = get_align_tr(
        neu_data, select_block=1, select_pos=-1, time_before=time_before
    )
    # sample position in the screen
    position = neu_data.position[mask_in]
    u_pos = np.unique(position, axis=0)

    if u_pos.shape[0] > 1:
        print("Position of the sample change during the session %s" % path)
        x_pos, y_pos = np.nan, np.nan
    else:
        x_pos, y_pos = u_pos[0][0][0], u_pos[0][0][1]
    # -----
    # Select durarion to analyze
    sp_in = sp_in[:, : time_before + 460 + 700]
    sp_out = sp_out[:, : time_before + 460 + 700]
    # get avg fr over trials and time
    avgfr_in = np.nan
    avgfr_out = np.nan
    if np.logical_and(sp_in.shape[0] > 2, sp_in.ndim > 1):
        avgfr_in = get_avg_fr(sp_in)
    if np.logical_and(sp_out.shape[0] > 2, sp_out.ndim > 1):
        avgfr_out = get_avg_fr(sp_out)
    # ----VD------
    vd_in, vd_out, bl_in, g1_in, g2_in, bl_out, g1_out, g2_out = compute_vd_idx(
        neu_data
    )
    # -------------
    # get fr
    sp_in = moving_average(data=sp_in, win=100, step=1)
    sp_out = moving_average(data=sp_out, win=100, step=1)
    # split sp by samples id

    samples = [11, 15, 51, 55, 0]
    for in_out, mask, sp in zip(["in", "out"], [mask_in, mask_out], [sp_in, sp_out]):
        sample_id = neu_data.sample_id[mask]
        sp_samples = get_sp_by_sample(sp, samples, sample_id)
        o1 = np.concatenate((sp_samples["11"], sp_samples["15"]))
        o5 = np.concatenate((sp_samples["51"], sp_samples["55"]))
        c1 = np.concatenate((sp_samples["11"], sp_samples["51"]))
        c5 = np.concatenate((sp_samples["15"], sp_samples["55"]))
        sample = np.concatenate(
            (sp_samples["11"], sp_samples["15"], sp_samples["51"], sp_samples["55"])
        )
        n0 = sp_samples["0"]
        # Check selectivity and latency
        color_lat, color_score = get_selectivity(c1, c5, win=win)
        color_selec = "c1" if color_score > 0 else "c5" if color_score < 0 else "nan"
        orient_lat, orient_score = get_selectivity(o1, o5, win=win)
        orient_selec = "o1" if orient_score > 0 else "o5" if orient_score < 0 else "nan"
        neutral_lat, neutral_score = get_selectivity(sample, n0, win=win)
        neutral_selec = (
            "sample" if neutral_score > 0 else "n0" if neutral_score < 0 else "nan"
        )
        # save results in Dict
        neu_info["color_lat_" + in_out] = color_lat
        neu_info["color_selec_" + in_out] = color_selec
        neu_info["orient_lat_" + in_out] = orient_lat
        neu_info["orient_selec_" + in_out] = orient_selec
        neu_info["neutral_lat_" + in_out] = neutral_lat
        neu_info["neutral_selec_" + in_out] = neutral_selec

    pos_lat, pos_score = get_selectivity(sp_in, sp_out, win=win)
    pos_selec = "in" if pos_score > 0 else "out" if pos_score < 0 else "nan"
    # extra info
    subject = path.rsplit("/")[-8]
    area = path.rsplit("/")[-3]
    session = path.rsplit("/")[-1].rsplit("_" + subject)[0]
    ch_start = np.load(
        "/envau/work/invibe/USERS/IBOS/openephys/Riesling/"
        + session
        + "/Record Node 102/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/KS"
        + area.upper()
        + "/channel_map.npy"
    )[0][0]
    cluster_ch = neu_data.cluster_ch - ch_start
    matrix_df = pd.read_csv(
        "/envau/work/invibe/USERS/LOSADA/Users/losadac/data_stats/"
        + area
        + "_ch_pos.csv",
        header=0,
        index_col=0,
    )
    matrix = matrix_df.values
    matrix = matrix - matrix.min().min()
    row, col = np.where(cluster_ch == matrix)
    # save results in Dict
    neu_info["path"] = path
    neu_info["session"] = session
    neu_info["cluster_ch"] = cluster_ch
    neu_info["pos_lat"] = pos_lat
    neu_info["pos_selec"] = pos_selec
    neu_info["avgfr_in"] = avgfr_in
    neu_info["avgfr_out"] = avgfr_out
    neu_info["vd_in"] = vd_in
    neu_info["vd_out"] = vd_out
    neu_info["bl_in"] = bl_in
    neu_info["s_in"] = g1_in
    neu_info["d_in"] = g2_in
    neu_info["bl_out"] = bl_out
    neu_info["s_out"] = g1_out
    neu_info["d_out"] = g2_out
    neu_info["t_before_s_on"] = time_before
    neu_info["area"] = area
    neu_info["matrix_row"] = row[0]
    neu_info["matrix_col"] = col[0]
    neu_info["cluster_group"] = neu_data.cluster_group
    neu_info["x_pos_sample"] = x_pos
    neu_info["y_pos_sample"] = y_pos
    return neu_info


# ----------------------------------------------------------------------------------------------#

#### Define parameters and compute neurons df
if platform.system() == "Linux":
    basepath = (
        "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/"
    )
elif platform.system() == "Windows":
    basepath = "C:/Users/camil/Documents/int/"


save_path = "./"
areas = ["pfc", "v4"]  # ,'pfc']

## read paths
paths_info = {"lip": {}, "v4": {}, "pfc": {}}
for area in areas:
    neu_path = basepath + "session_struct/" + area + "/neurons/*neu.h5"
    path_list = glob.glob(neu_path)
    paths_info[area]["paths"] = path_list
    pp = []
    for path in path_list[:300]:
        pp.append(path.rsplit("/")[-1].rsplit("_Riesling")[0])
    paths_info[area]["sessions"] = np.unique(pp).tolist()

## main computation
area_info = {}
for area in areas:
    all_paths = paths_info[area]["paths"]
    info = Parallel(n_jobs=-1)(
        delayed(get_neuron_info)(all_paths[i]) for i in tqdm(range(len(all_paths)))
    )
    area_info[area] = info
    df_keys = list(area_info[area][0].keys())
    df_aux: Dict[str, list] = defaultdict(list)
    for i in range(len(area_info[area])):
        for key in df_keys:
            df_aux[key] += [area_info[area][i][key]]
    areas_df = pd.DataFrame(df_aux)
    file_name = save_path + area + "_neurons_info.csv"
    areas_df.to_csv(file_name, index=False)
