from ephysvibe.structures.neuron_data import NeuronData
import numpy as np
from sklearn.decomposition import PCA, SparsePCA
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

seed = 2024


def compute_pca(x, n_comp=50):
    model = PCA(n_components=n_comp).fit(x.T)
    C = model.components_
    pc_s = C @ x
    return model, pc_s


def compute_sparse_pca(x, n_comp=50):
    model = SparsePCA(n_components=n_comp).fit(x.T)
    C = model.components_
    pc_s = C @ x
    return model, pc_s


def plot_pc(pcomp, colors, t_epochs, area, figsize):
    fig, ax = plt.subplots(2, 3, figsize=figsize)
    j_ax = 0
    i_ax = 0
    for i in np.arange(0, 4):
        for j in np.arange(0 + i, 4):
            if i == j:
                continue
            if j_ax >= 3:
                i_ax = 1
                j_ax = 0
            for key in t_epochs.keys():
                ax[i_ax, j_ax].scatter(
                    pcomp[i][t_epochs[key]],
                    pcomp[j][t_epochs[key]],
                    s=8,
                    color=colors[key],
                    label=key,
                )

            ax[i_ax, j_ax].set(xlabel="PC " + str(i + 1), ylabel="PC " + str(j + 1))
            j_ax += 1
    fig.suptitle(area)
    ax[0, 0].legend(
        fontsize=10, scatterpoints=5, columnspacing=0.5, framealpha=0, loc="best"
    )
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.8)


def plot_pc_neu(pcomp, pcomp_neu, colors, colors_neu, t_epochs, area, figsize):
    fig, ax = plt.subplots(2, 3, figsize=figsize)
    j_ax = 0
    i_ax = 0
    for i in np.arange(0, 4):
        for j in np.arange(0 + i, 4):
            if i == j:
                continue
            if j_ax >= 3:
                i_ax = 1
                j_ax = 0
            for key in t_epochs.keys():
                ax[i_ax, j_ax].scatter(
                    pcomp[i][t_epochs[key]],
                    pcomp[j][t_epochs[key]],
                    s=8,
                    color=colors[key],
                    label=key,
                )
                ax[i_ax, j_ax].scatter(
                    pcomp_neu[i][t_epochs[key]],
                    pcomp_neu[j][t_epochs[key]],
                    s=8,
                    color=colors_neu[key],
                    label=key,
                )
            ax[i_ax, j_ax].set(xlabel="PC " + str(i + 1), ylabel="PC " + str(j + 1))
            j_ax += 1
    fig.suptitle(area)
    ax[0, 0].legend(
        fontsize=10, scatterpoints=5, columnspacing=0.5, framealpha=0, loc="best"
    )
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.8)


def plot_pc_3d(pcomp, colors, t_epochs, area, figsize=(5, 5)):
    i = 0
    fig, ax = plt.subplots(
        figsize=figsize, sharey=True, sharex=True, subplot_kw={"projection": "3d"}
    )
    for key in t_epochs.keys():
        ax.plot(
            pcomp[i][t_epochs[key]],
            pcomp[i + 1][t_epochs[key]],
            pcomp[i + 2][t_epochs[key]],
            color=colors[key],
            label=key,
        )
    fig.suptitle(area)
    ax.set(xlabel="PC " + str(i), ylabel="PC " + str(i + 1), zlabel="PC " + str(i + 2))
    ax.legend(
        fontsize=9, columnspacing=0.5, facecolor="white", framealpha=1, loc="best"
    )
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.8)


def plot_explained_var(model, figsize, area):
    fig, ax = plt.subplots(figsize=figsize)
    exp_var_pca = model.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    # Create the visualization plot
    ax.bar(
        range(0, len(exp_var_pca)),
        exp_var_pca,
        alpha=0.5,
        align="center",
        label="Individual explained variance",
    )
    ax.step(
        range(0, len(cum_sum_eigenvalues)),
        cum_sum_eigenvalues,
        where="mid",
        label="Cumulative explained variance",
    )
    ax.set(
        xlabel="Principal component index",
        ylabel="Explained variance ratio",
        title=area,
    )
    ax.legend(loc="best")
    print(
        "%s: %d components to explain 80%% of the variance"
        % (area, np.where(cum_sum_eigenvalues > 0.8)[0][0])
    )
