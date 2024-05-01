import numpy as np
from scipy import stats
from ephysvibe.trials import align_trials, select_trials
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.stats import smetrics
from collections import defaultdict
from typing import Dict
from ephysvibe.structures.neuron_data import NeuronData
import pandas as pd
import platform
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json
import os
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def moving_average(data: np.ndarray, win: int, step: int = 1) -> np.ndarray:
    d_shape = data.shape
    count = 0
    win = int(np.floor(win / 2))
    if len(d_shape) == 3:
        d_avg = np.zeros((d_shape[0], d_shape[1], int(np.floor(d_shape[2] / step))))
        for i_step in np.arange(0, d_shape[2] - step, step):
            st_win = 0 if i_step - win < 0 else i_step - win
            d_avg[:, :, count] = np.mean(data[:, :, st_win : i_step + win], axis=2)
            count += 1
    if len(d_shape) == 2:
        d_avg = np.zeros((d_shape[0], int(np.floor(d_shape[1] / step))))
        for i_step in np.arange(0, d_shape[1] - step, step):
            st_win = 0 if i_step - win < 0 else i_step - win
            d_avg[:, count] = np.mean(data[:, st_win : i_step + win], axis=1)
            count += 1
    if len(d_shape) == 1:
        d_avg = np.zeros((int(np.floor(d_shape[0] / step))))
        for i_step in np.arange(0, d_shape[0] - step, step):
            st_win = 0 if i_step - win < 0 else i_step - win
            d_avg[count] = np.mean(data[st_win : i_step + win], axis=0)
            count += 1
    return d_avg


def get_visual_fr(sp, start, st_v=50, end_v=250):
    if np.nansum(sp[:, : start + end_v]) == 0:
        return np.nan, np.nan
    avgtr_fr = np.mean(sp, axis=0)
    bl_avg = np.mean(avgtr_fr[:start])
    avg_fr = avgtr_fr - bl_avg
    vr = np.mean(avg_fr[start + st_v : start + end_v] * 1000)  # vis response

    g1 = np.mean(sp[:, :start], axis=1)
    g2 = np.mean(sp[:, start + st_v : start + end_v], axis=1)
    p = stats.ranksums(g1, g2)[1]
    sig = p < 0.5
    return vr, sig


def main(
    path: Path,
    sessions_path: Path,
    ch_pos_path: Path,
    time_before: int = 500,
    start: int = -200,
    end: int = 1000,
    mov_avg_win: int = 100,
    selec_win: int = 75,
    st_v: int = 50,
    end_v: int = 200,
    st_d: int = 100,
    end_d: int = 300,
    vd_pwin: int = 150,
    vd_avg_win: int = 200,
) -> Dict:
    neu_info = {}
    idx_start = time_before + start
    idx_end = time_before + end
    # Read neuron data
    neu_data = NeuronData.from_python_hdf5(path)

    # get spike matrices in and out conditions
    sp_in, mask_in = align_trials.get_align_tr(
        neu_data, select_block=1, select_pos=1, time_before=time_before
    )
    sp_out, mask_out = align_trials.get_align_tr(
        neu_data, select_block=1, select_pos=-1, time_before=time_before
    )
    sp_in_d, _ = align_trials.get_align_tr(
        neu_data,
        select_block=1,
        select_pos=1,
        time_before=time_before,
        event="sample_off",
    )
    sp_out_d, _ = align_trials.get_align_tr(
        neu_data,
        select_block=1,
        select_pos=-1,
        time_before=time_before,
        event="sample_off",
    )
    # sample position in the screen
    position = neu_data.position[mask_in]
    u_pos = np.unique(position, axis=0)

    if u_pos.shape[0] > 1:
        x_pos, y_pos = np.nan, np.nan
    else:
        x_pos, y_pos = u_pos[0][0][0], u_pos[0][0][1]
    # -----

    # get avg fr over trials and time
    avgfr_in = np.nan
    avgfr_out = np.nan
    if np.logical_and(sp_in.shape[0] > 2, sp_in.ndim > 1):
        avgfr_in = np.nanmean(sp_in[:, idx_start:idx_end]) * 1000
    if np.logical_and(sp_out.shape[0] > 2, sp_out.ndim > 1):
        avgfr_out = np.nanmean(sp_out[:, idx_start:idx_end]) * 1000
    # ----VD------
    vd_in, bl_in, g1_in, g2_in = smetrics.compute_vd_idx(
        neu_data=neu_data,
        time_before=abs(start),
        st_v=st_v,
        end_v=end_v,
        st_d=st_d,
        end_d=end_d,
        vd_pwin=vd_pwin,
        vd_avg_win=vd_avg_win,
        in_out=1,
    )
    vd_out, bl_out, g1_out, g2_out = smetrics.compute_vd_idx(
        neu_data=neu_data,
        time_before=abs(start),
        st_v=st_v,
        end_v=end_v,
        st_d=st_d,
        end_d=end_d,
        vd_pwin=vd_pwin,
        vd_avg_win=vd_avg_win,
        in_out=-1,
    )
    # -------------
    # Select durarion to analyze
    # sp_in = sp_in[:, : idx_end + mov_avg_win]
    # sp_out = sp_out[:, : idx_end + mov_avg_win]
    # get fr
    sp_in = firing_rate.moving_average(data=sp_in, win=mov_avg_win, step=1)[
        :, idx_start:idx_end
    ]
    sp_out = firing_rate.moving_average(data=sp_out, win=mov_avg_win, step=1)[
        :, idx_start:idx_end
    ]
    sp_in_d = firing_rate.moving_average(data=sp_in_d, win=mov_avg_win, step=1)[
        :, idx_start:idx_end
    ]
    sp_out_d = firing_rate.moving_average(data=sp_out_d, win=mov_avg_win, step=1)[
        :, idx_start:idx_end
    ]
    # ----- Visual response
    sp_vr_in = sp_in[neu_data.sample_id[mask_in] != 0]
    vr_in, vr_in_sig = get_visual_fr(
        sp=sp_vr_in, start=abs(start), st_v=st_v, end_v=end_v
    )
    sp_vr_out = sp_out[neu_data.sample_id[mask_out] != 0]
    vr_out, vr_out_sig = get_visual_fr(
        sp=sp_vr_out, start=abs(start), st_v=st_v, end_v=end_v
    )
    neu_info["vr_in"] = vr_in
    neu_info["vr_in_sig"] = vr_in_sig
    neu_info["vr_out"] = vr_out
    neu_info["vr_out_sig"] = vr_out_sig
    # ----- Analysis by sample (selectivity, vd)
    samples = [11, 15, 51, 55, 0]
    for in_out, mask, sp in zip(
        ["in", "out"], [mask_in, mask_out], [[sp_in, sp_in_d], [sp_out, sp_out_d]]
    ):
        # split sp by samples id
        sample_id = neu_data.sample_id[mask]
        sp_samples = select_trials.get_sp_by_sample(sp[0], sample_id, samples)
        o1 = np.concatenate((sp_samples["11"], sp_samples["15"]))
        o5 = np.concatenate((sp_samples["51"], sp_samples["55"]))
        c1 = np.concatenate((sp_samples["11"], sp_samples["51"]))
        c5 = np.concatenate((sp_samples["15"], sp_samples["55"]))
        sample = np.concatenate(
            (sp_samples["11"], sp_samples["15"], sp_samples["51"], sp_samples["55"])
        )
        n0 = sp_samples["0"]
        # Check selectivity and latency
        color_lat, color_score = smetrics.get_selectivity(c1, c5, win=selec_win)
        color_selec = "c1" if color_score > 0 else "c5" if color_score < 0 else "nan"
        orient_lat, orient_score = smetrics.get_selectivity(o1, o5, win=selec_win)
        orient_selec = "o1" if orient_score > 0 else "o5" if orient_score < 0 else "nan"
        neutral_lat, neutral_score = smetrics.get_selectivity(sample, n0, win=selec_win)
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
        ## VD index per sample
        sample_id = neu_data.sample_id[mask]
        sp_samples_d = select_trials.get_sp_by_sample(sp[1], sample_id, samples)
        o1_d = np.concatenate((sp_samples_d["11"], sp_samples_d["15"]))
        o5_d = np.concatenate((sp_samples_d["51"], sp_samples_d["55"]))
        c1_d = np.concatenate((sp_samples_d["11"], sp_samples_d["51"]))
        c5_d = np.concatenate((sp_samples_d["15"], sp_samples_d["55"]))

        for i_sam, isp_s, isp_d in zip(
            ["o1", "o5", "c1", "c5"], [o1, o5, c1, c5], [o1_d, o5_d, c1_d, c5_d]
        ):
            vd_idx, bl_mean, g1_mean, g2_mean = smetrics.compute_vd_idx(
                sp_s=isp_s,
                sp_d=isp_d,
                time_before=abs(start),
                st_v=st_v,
                end_v=end_v,
                st_d=st_d,
                end_d=end_d,
                vd_pwin=vd_pwin,
                vd_avg_win=vd_avg_win,
            )
            neu_info[i_sam + "_vd_" + in_out] = vd_idx
            neu_info[i_sam + "_bl_" + in_out] = bl_mean
            neu_info[i_sam + "_s_" + in_out] = g1_mean
            neu_info[i_sam + "_d_" + in_out] = g2_mean

    pos_lat, pos_score = smetrics.get_selectivity(sp_in, sp_out, win=selec_win)
    pos_selec = "in" if pos_score > 0 else "out" if pos_score < 0 else "nan"
    # extra info
    subject = path.rsplit("/")[-8]
    area = path.rsplit("/")[-3]
    session = path.rsplit("/")[-1].rsplit("_" + subject)[0]
    ch_start = np.load(
        sessions_path
        + session
        + "/Record Node 102/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/KS"
        + area.upper()
        + "/channel_map.npy"
    )[0][0]
    cluster_ch = neu_data.cluster_ch - ch_start
    matrix_df = pd.read_csv(
        ch_pos_path + area + "_ch_pos.csv",
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
    neu_info["area"] = area
    neu_info["matrix_row"] = row[0]
    neu_info["matrix_col"] = col[0]
    neu_info["cluster_group"] = neu_data.cluster_group
    neu_info["x_pos_sample"] = x_pos
    neu_info["y_pos_sample"] = y_pos
    return neu_info


# ----------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="Path to neuron data (neu.h5)", type=Path)
    parser.add_argument("--sessions_path", help="", type=Path)
    parser.add_argument("--ch_pos_path", help="", type=Path)
    parser.add_argument("--time_before", help="", type=int)
    parser.add_argument("--start", help="", type=int)
    parser.add_argument("--end", help="", type=int)
    parser.add_argument("--mov_avg_win", help="", type=int)
    parser.add_argument("--selec_win", help="", type=int)
    parser.add_argument("--st_v", type=int)
    parser.add_argument("--end_v", type=int)
    parser.add_argument("--st_d", type=int)
    parser.add_argument("--end_d", type=int)
    parser.add_argument("--vd_pwin", help="", type=int)
    parser.add_argument("--vd_avg_win", help="", type=int)
    args = parser.parse_args()

    #### Define parameters and compute neurons df

    try:
        main(
            path=args.path,
            sessions_path=args.sessions_path,
            ch_pos_path=args.ch_pos_path,
            time_before=args.time_before,
            start=args.start,
            end=args.end,
            mov_avg_win=args.mov_avg_win,
            selec_win=args.selec_win,
            st_v=args.st_v,
            end_v=args.end_v,
            st_d=args.st_d,
            end_d=args.end_d,
            vd_pwin=args.vd_pwin,
            vd_avg_win=args.vd_avg_win,
        )
    except FileExistsError:
        logging.error("filepath does not exist")
