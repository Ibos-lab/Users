import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal
from ephysvibe.trials import select_trials, align_trials
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.task import task_constants
import glm_functions
import os
import platform
from joblib import Parallel, delayed
from tqdm import tqdm

# import structures
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.structures.eye_data import EyeData
from ephysvibe.structures.spike_data import SpikeData
from ephysvibe.structures.bhv_data import BhvData

# Torch
import torch
import torch.nn as nn
import torch.distributions as D
import torch.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_poisson_deviance
import nn_model
from sklearn.model_selection import KFold

import logging

seed = 2024


def train_model(
    model,
    X_ten_train,
    y_ten_train,
    X_ten_test,
    y_ten_test,
    epochs,
    optimizer,
    criterion,
    early_stopping,
    regl1=None,
    regl2=None,
):
    losses_train, losses_test, acc_test, acc_train, allit_w = [], [], [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg_loss1, reg_loss2 = 0, 0
    regularizer1, regularizer2 = None, None
    if regl1 != None and regl2 != None:
        regularizer1 = regl1
        regularizer2 = regl2
    elif regl1 != None:
        regularizer1 = regl1
    elif regl2 != None:
        regularizer2 = regl2

    for _ in range(epochs):
        model.train()
        batch_loss = 0
        batch_acc = 0
        # for X_loader, y_loader in train_loader:
        # X_loader,y_loader=X_loader.to(device),y_loader.to(device)
        optimizer.zero_grad()  # zero out gradients
        y_pred_train = model(X_ten_train)  # fordward propagation
        loss_train = criterion(y_pred_train, y_ten_train)  # computing the loss
        if regularizer1 != None:
            reg_loss1 = regularizer1(model)
        if regularizer2 != None:
            reg_loss2 = regularizer2(model)
        loss_train = loss_train + reg_loss1 + reg_loss2
        loss_train.backward()  # backpropagation
        # batch_loss+= loss_train.item()
        acc = mean_poisson_deviance(
            y_ten_train.cpu().detach().numpy().reshape(-1),
            y_pred_train[:, 0].cpu().detach().numpy(),
        )
        # batch_acc+=acc
        optimizer.step()  # optimize (GD)
        losses_train.append(loss_train.cpu().detach().numpy().item())
        acc_train.append(acc)
        model.eval()
        y_pred_test = model(X_ten_test)
        loss_test = criterion(y_pred_test, y_ten_test)
        losses_test.append(loss_test.cpu().detach().numpy().item())
        acc_test.append(
            mean_poisson_deviance(
                y_ten_test.cpu().detach().numpy(),
                y_pred_test[:, 0].cpu().detach().numpy(),
            )
        )
        # Check early stopping criteria
        early_stopping(loss_train, model)
        if early_stopping.early_stop:
            # print("Early stopping")
            break
    return losses_train, losses_test, acc_train, acc_test, model


neu_path = ""
neu_data = NeuronData.from_python_hdf5(neu_path)

# parameters
time_before = 0
select_block = 1

events = [
    "hist",
    "s_o1",
    "s_o5",
    "s_c1",
    "s_c5",
    "s_n0",
    "d_o1",
    "d_o5",
    "d_c1",
    "d_c5",
    "d_n0",
]
last_event = "test_on_1"

# Select trials aligned to
code = 1
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
# -------- basis -------------
# Number of basis for each stimulus
fix_cue_dim = 4
fixation_dim = 5
sample_dim = 40
delay_dim = 40
test_dim = 20
match_dim = 10
bar_dim = 10
hist_cos_dim = 15
hist_box_dim = 10
hist_dim = hist_cos_dim + hist_box_dim
# Compute basis
cos_basis = glm_functions.make_non_linear_raised_cos(
    nBases=hist_cos_dim, binSize=1, endPoints=np.array([10, 150]), nlOffset=1
)
single_bin_boxcars = np.zeros((cos_basis.shape[0], hist_box_dim))
single_bin_boxcars[range(hist_box_dim), range(hist_box_dim)] = 1
hist_b = np.concatenate((single_bin_boxcars, cos_basis), axis=1)
fix_cue_b = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=fix_cue_dim, duration=500
)
fixation_b = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=fixation_dim, duration=1200
)
sample_b = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=sample_dim, duration=400
)
delay_b = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=delay_dim, duration=400
)
test_b = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=test_dim, duration=400
)
match_b = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=match_dim, duration=200
)
bar_release_b = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=bar_dim, duration=80
)
# create dict with the basis
basis = {
    "fix_cue": fix_cue_b,
    "fixation": fixation_b,
    "hist": hist_b,
    "s_on_off": [np.nan],
    "s_o1": sample_b,
    "s_o5": sample_b,
    "s_c1": sample_b,
    "s_c5": sample_b,
    "s_n0": sample_b,
    "d_o1": delay_b,
    "d_o5": delay_b,
    "d_c1": delay_b,
    "d_c5": delay_b,
    "d_n0": delay_b,
    "d_on_off": [np.nan],
}
for i in range(1, 9):
    basis["test_color" + str(i)] = test_b
    basis["test_orient" + str(i)] = test_b
for i in range(1, 5):
    basis["test_on_off" + str(i)] = [np.nan]
basis["match"] = match_b
basis["bar_release"] = bar_release_b

# -------- end -------------

# select trials index
sample_id = neu_data.sample_id[mask]
# select trials per sample
sn0_tr = np.where(sample_id == 0)[0]
s15_tr = np.where(sample_id == 15)[0]
s11_tr = np.where(sample_id == 11)[0]
s55_tr = np.where(sample_id == 55)[0]
s51_tr = np.where(sample_id == 51)[0]
# Shuffle trials
rng = np.random.default_rng(seed=seed)
rng.shuffle(sn0_tr)
rng.shuffle(s15_tr)
rng.shuffle(s11_tr)
rng.shuffle(s55_tr)
rng.shuffle(s51_tr)

n_is = 15
st_tt = n_is * 5
# Select trials for testing
sn0 = sn0_tr[:n_is]
s15 = s15_tr[:n_is]
s11 = s11_tr[:n_is]
s55 = s55_tr[:n_is]
s51 = s51_tr[:n_is]
idxs_trials = np.concatenate((sn0, s15, s11, s55, s51))

# --- preprocess data
design_mat = []
all_dm = []
all_len_tr = []
all_s_on = []
all_fix_on = []
all_stim = []
all_sp = []
all_d_on = []
all_test1_on = []
all_sample_id = []
n_tr, _ = sp_sample_on.shape
data = Parallel(n_jobs=-1)(
    delayed(glm_functions.get_dm)(
        i_tr, last_event, events, neu_data, time_before, sp_sample_on, mask, basis
    )
    for i_tr in tqdm(idxs_trials)
)
for asc in data:
    design_mat.append(asc["design_mat"])
    all_dm.append(asc["all_dm"])
    all_len_tr.append(asc["len_tr"])
    all_s_on.append(asc["s_on"])
    all_fix_on.append(asc["fix_on"])
    all_d_on.append(asc["delay_on"])
    all_stim.append(asc["stim"])
    all_sp.append(asc["sp"])
    all_test1_on.append(asc["test1_on"])
    all_sample_id.append(int(asc["sample_id"]))
indices = data[0]["indices"]
# --- end preprocess data

# ---Select trials for training and validation---
sn0_pos = np.where(np.array(all_sample_id) == 0)[0]
s15_pos = np.where(np.array(all_sample_id) == 15)[0]
s11_pos = np.where(np.array(all_sample_id) == 11)[0]
s55_pos = np.where(np.array(all_sample_id) == 55)[0]
s51_pos = np.where(np.array(all_sample_id) == 51)[0]
idx_dict = {
    "sn0": sn0_pos,
    "s15": s15_pos,
    "s11": s11_pos,
    "s55": s55_pos,
    "s51": s51_pos,
}

kf = KFold(n_splits=5, shuffle=False)
kfolds = list(kf.split(np.arange(n_is)))
ki_train, ki_val = kfolds[1]
idxs_train, idxs_val = nn_model.get_idx_trials_train_val(
    ki_train, ki_val, idx_dict, seed
)

# train
n_train = len(idxs_train)
sp_train = np.concatenate(list(all_sp[i] for i in idxs_train.tolist()))
logging.info("Length of all trials concatenated: %s" % sp_train.shape)
dm_train = np.concatenate(list(design_mat[i] for i in idxs_train.tolist()))
logging.info("Shape of dm: (%s, %s)" % dm_train.shape)
len_tr_train = np.array(all_len_tr)[idxs_train]
logging.info("Number of trials %s" % len_tr_train.shape)
s_on_train = np.array(all_s_on)[idxs_train]
fix_on_train = np.array(all_fix_on)[idxs_train]
d_on_train = np.array(all_d_on)[idxs_train]
test1_on_train = np.array(all_test1_on)[idxs_train]
# validation
sp_test = np.concatenate(list(all_sp[i] for i in idxs_val.tolist()))
logging.info("Length of all trials concatenated: %s" % sp_test.shape)
dm_test = np.concatenate(list(design_mat[i] for i in idxs_val.tolist()))
logging.info("Shape of dm: (%s, %s)" % dm_test.shape)
len_tr_test = np.array(all_len_tr)[idxs_val]
logging.info("Number of trials %s" % len_tr_test.shape)
s_on_test = np.array(all_s_on)[idxs_val]
fix_on_test = np.array(all_fix_on)[idxs_val]
d_on_test = np.array(all_d_on)[idxs_val]
test1_on_test = np.array(all_test1_on)[idxs_val]

# --- End select trials for training and validation---

# -- Nomalization
x = dm_train
mu = np.mean(x, axis=0)
sigma = np.std(x, axis=0)
sigma0 = sigma
sigma0[sigma0 == 0] = 1
dm_train = (x - mu) / sigma0

x_test = dm_test
mu = np.mean(x_test, axis=0)
sigma = np.std(x_test, axis=0)
sigma0 = sigma
sigma0[sigma0 == 0] = 1
dm_test = (x_test - mu) / sigma0
# --

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Using device:", device)

# Split data
X_train = dm_train  # design_mat_offset#np.random.rand(1000, 1).astype(np.float32)  #
y_train = sp_train[
    :, np.newaxis
]  # np.exp(2 * X) + np.random.normal(0, 0.1, size=(1000, 1)).astype(np.float32)  #  rng.poisson(1,size=sp_train.shape[0])[:,np.newaxis] #
X_test = dm_test
y_test = sp_test[
    :, np.newaxis
]  # .copy() # rng.integers(low=0, high=2, size=sp_test.shape[0], dtype=int)[:,np.newaxis]#sp_test  sp_test.shape[0]
# Convert data to PyTorch tensors
X_ten_train = torch.from_numpy(X_train).to(torch.float32).to(device)
y_ten_train = torch.from_numpy(y_train).to(torch.int).to(device)
X_ten_test = torch.from_numpy(X_test).to(torch.float32).to(device)
y_ten_test = torch.from_numpy(y_test).to(torch.int).to(device)

# --- Train model

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
lambdasl1 = np.arange(0, 1, 1 / 20000)[:30]  # np.concatenate(([0],2**np.arange(20)))
lambdasl2 = np.arange(0, 1, 1 / 10000)
lambdas = np.arange(0, 1, 1 / 100000)[np.arange(0, 11000, 100)][
    :30
]  # np.concatenate(([0],(2**np.arange(30))/100000000))
all_losses_train, all_losses_test, all_acc_test, all_acc_train = {}, {}, {}, {}
f_l, ax_l = plt.subplots()
f_a, ax_a = plt.subplots()
all_models = {}
lambdas = [
    0,
    0.000005,
    0.00001,
    0.00005,
    0.0001,
    0.0006,
    0.0009,
    0.001,
    0.005,
    0.01,
    0.03,
    0.05,
    0.09,
]
for il, lam in enumerate(lambdas):
    torch.manual_seed(seed)
    glm_model = nn_model.ExpModel(input_dim, output_dim).to(device)
    # Define the loss function and optimizer
    criterion = nn_model.poisson_glm_loss  # nn.PoissonNLLLoss(False)#
    optimizer = torch.optim.Adam(glm_model.parameters(), lr=0.01)  # , weight_decay=lam)
    # Train the model
    epochs = 10000
    early_stopping = nn_model.EarlyStopping(patience=1000, verbose=True, delta=0.00001)
    regl1 = None  # nn_model.L1Regularization(beta=0.00005)#lambdasl1[2])#
    regl2 = nn_model.SmoothL2Regularization(beta=lam)
    # regularizer = SmoothL2Regularization(beta=lam)
    losses_train, losses_test, acc_train, acc_test, trained_model = train_model(
        glm_model,
        X_ten_train,
        y_ten_train,
        X_ten_test,
        y_ten_test,
        epochs,
        optimizer,
        criterion,
        early_stopping,
        regl1=regl1,
        regl2=regl2,
    )
    all_models[str(lam)] = trained_model.cpu()  # save trained models in a dict

    all_losses_train[str(lam)] = losses_train
    all_losses_test[str(lam)] = losses_test
    all_acc_train[str(lam)] = acc_train
    all_acc_test[str(lam)] = acc_test

# --- End train model

## Plot resulst
lam_min = 1
i_min = 0
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
for i in range(0, len(lambdas)):
    y = np.array(all_losses_train[str(lambdas[i])])[-1]
    ax1.scatter(i, y, c="r")
    y = np.array(all_acc_train[str(lambdas[i])])[-1]
    ax2.scatter(i, y, c="r")
    y = np.array(all_losses_test[str(lambdas[i])])[-1]
    if i != 0:
        if y < lam_min:
            lam_min = y
            i_min = i

    ax3.scatter(i, y, c="g")
    y = np.array(all_acc_test[str(lambdas[i])])[-1]
    ax4.scatter(i, y, c="g")
ax1.set_title("Train Loss")
ax2.set_title("Train Deviance")
ax3.set_title("Validation Loss")
ax4.set_title("Validation Deviance")
ax1.vlines(i_min, ax1.get_ylim()[0], ax1.get_ylim()[1])
ax3.vlines(i_min, ax3.get_ylim()[0], ax3.get_ylim()[1])
ax3.set_xlabel(lambdas)
