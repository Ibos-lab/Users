import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import platform
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.linalg import block_diag

# ephys funcs
from ephysvibe.trials import align_trials
from ephysvibe.task import task_constants
from ephysvibe.trials.spikes import firing_rate

# import structures
from ephysvibe.structures.neuron_data import NeuronData

# parallel
from joblib import Parallel, delayed
from tqdm import tqdm
import glm_functions
import logging

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)
# parameters
neuronpath = (
    "/session_struct/lip/neurons/2023-11-30_11-16-24_Riesling_lip_e1_r1_good1_neu.h5"
)
outputpath = "/envau/work/invibe/USERS/LOSADA/Users/losadac/results/"  # "/home/INT/losada.c/Documents/codes/results"
time_before = 0
select_block = 1
train_perc = 0.80
events = [
    "fix_cue",
    "fixation",
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
    "test_orient1",
    "test_orient2",
    "test_orient3",
    "test_orient4",
    "test_orient5",
    "test_orient6",
    "test_orient7",
    "test_orient8",
    "test_color1",
    "test_color2",
    "test_color3",
    "test_color4",
    "test_color5",
    "test_color6",
    "test_color7",
    "test_color8",
    "match",
    "bar_release",
]
last_event = "reward"

if platform.system() == "Linux":
    basepath = (
        "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/"
    )
elif platform.system() == "Windows":
    basepath = "C:/Users/camil/Documents/int/"

# Load data
neu_path = basepath + neuronpath
neu_data = NeuronData.from_python_hdf5(neu_path)

# Select trials aligned to start_trial
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
    event="start_trial",
    time_before=time_before,
    error_type=0,
)

## Define basis
# Number of basis for each stimulus
fix_cue_dim = 4
fixation_dim = 5
sample_dim = 10
delay_dim = 10
test_dim = 10
match_dim = 4
bar_dim = 4
hist_cos_dim = 10
hist_box_dim = 10
hist_dim = hist_cos_dim + hist_box_dim
# Compute basis
cos_basis = glm_functions.make_non_linear_raised_cos(
    nBases=hist_cos_dim, binSize=1, endPoints=np.array([10, 150]), nlOffset=1
)
single_bin_boxcars = np.zeros((cos_basis.shape[0], hist_box_dim))
single_bin_boxcars[range(hist_box_dim), range(hist_box_dim)] = 1
hist_basis = np.concatenate((single_bin_boxcars, cos_basis), axis=1)
fix_cue_basis = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=fix_cue_dim, duration=500
)
fixation_basis = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=fixation_dim, duration=1200
)
sample_basis = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=sample_dim, duration=400
)
delay_basis = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=delay_dim, duration=400
)
test_basis = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=test_dim, duration=400
)
match_basis = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=match_dim, duration=400
)
bar_release_basis = glm_functions.compute_smooth_temporal_basis(
    btype="raised cosine", nBases=bar_dim, duration=80
)
basis = {
    "fix_cue": fix_cue_basis,
    "fixation": fixation_basis,
    "hist": hist_basis,
    "s_on_off": [np.nan],
    "s_o1": sample_basis,
    "s_o5": sample_basis,
    "s_c1": sample_basis,
    "s_c5": sample_basis,
    "s_n0": sample_basis,
    "d_o1": delay_basis,
    "d_o5": delay_basis,
    "d_c1": delay_basis,
    "d_c5": delay_basis,
    "d_n0": delay_basis,
    "d_on_off": [np.nan],
}
for i in range(1, 9):
    basis["test_color" + str(i)] = test_basis
    basis["test_orient" + str(i)] = test_basis
for i in range(1, 5):
    basis["test_on_off" + str(i)] = [np.nan]
basis["match"] = match_basis
basis["bar_release"] = bar_release_basis

# print filters and save
basis_keys = ["fix_cue", "hist", "s_o1", "d_o1", "test_color1", "match", "bar_release"]
cols = 2
rows = len(basis_keys) // cols
rows = rows + 1 if len(basis_keys) % cols > 0 else rows
f, ax = plt.subplots(rows, cols, figsize=(30, 10), sharey=True, sharex=True)
for iax, var in zip(np.concatenate(ax), basis_keys):
    iax.plot(basis[var])
    iax.set_title(var)
f.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.8)
f.savefig(outputpath + "basis.svg", format="svg", bbox_inches="tight")


# iterate over trials
def get_dm(i_tr):
    print(i_tr)
    e_time = neu_data.code_samples[mask][i_tr] - neu_data.code_samples[mask][i_tr][0]
    e_code = neu_data.code_numbers[mask][i_tr]
    test_stimuli = neu_data.test_stimuli[mask][i_tr]
    # n_test=np.sum(~np.isnan(test_stimuli))
    reward = task_constants.EVENTS_B1[last_event]
    idx_last = np.where(e_code == reward)[0]

    len_tr = int(e_time[idx_last][0] + 100)
    sample_id = neu_data.sample_id[mask][i_tr]
    stim = glm_functions.def_stim(
        events,
        len_tr,
        e_time,
        e_code,
        sample_id,
        sp_sample_on[i_tr, :len_tr],
        test_stimuli,
    )

    all_dm = {}
    indices = {}
    design_mat = np.zeros((len_tr, 1))
    shape = 0
    for key in events:
        base = basis[key]
        if np.any(np.isnan(base)):
            all_dm[key] = stim[key][:, np.newaxis]
        else:
            all_dm[key] = signal.convolve2d(stim[key][:, np.newaxis], base)[:len_tr]
        indices[key] = [shape, shape + all_dm[key].shape[1]]
        shape = shape + all_dm[key].shape[1]
        design_mat = np.concatenate((design_mat, all_dm[key]), axis=1)

    return {
        "design_mat": design_mat[:, 1:],
        "all_dm": all_dm,
        "stim": stim,
        "len_tr": len_tr,
        "s_on": e_time[4],
        "fix_on": e_time[2],
        "delay_on": e_time[5],
        "test1_on": e_time[6],
        "indices": indices,
        "sp": sp_sample_on[i_tr, :len_tr],
    }


sample_id = neu_data.sample_id[mask]
sn0 = np.where(sample_id == 0)[0][:30]
s15 = np.where(sample_id == 15)[0][:30]
s11 = np.where(sample_id == 11)[0][:30]
s55 = np.where(sample_id == 55)[0][:30]
s51 = np.where(sample_id == 51)[0][:30]
idxs = np.concatenate((sn0, s15, s11, s55, s51))
rng = np.random.default_rng(seed=2024)
rng.shuffle(idxs)

n_tr, _ = sp_sample_on.shape
data = Parallel(n_jobs=-1)(delayed(get_dm)(i_tr) for i_tr in tqdm(idxs))

design_mat = []
all_dm = []
all_len_tr = []
all_s_on = []
all_fix_on = []
all_stim = []
all_sp = []
all_d_on = []
all_test1_on = []
for asc in data:
    design_mat.append(asc["design_mat"])
    all_dm.append(asc["all_dm"])
    all_len_tr.append(asc["len_tr"])
    all_s_on.append(asc["s_on"])
    all_fix_on.append(asc["fix_on"])
    all_d_on.append(asc["delay_on"])
    all_stim.append(asc["stim"])
    all_sp.append(asc["stim"]["hist"][1:])
    all_test1_on.append(asc["test1_on"])
indices = data[0]["indices"]


# Split in train and test
n_train_tr = len(idxs)  # int(n_tr * train_perc)
logging.info("Number of trials for training %d" % n_train_tr)
# train
sp_train = np.concatenate(all_sp[:n_train_tr])
print("Length of all trials concatenated: %s" % sp_train.shape)
dm_train = np.concatenate(design_mat[:n_train_tr])
print("Shape of dm: (%s, %s)" % dm_train.shape)
len_tr_train = np.array(all_len_tr)[:n_train_tr]
print("Number of trials %s" % len_tr_train.shape)
s_on_train = np.array(all_s_on)[:n_train_tr]
fix_on_train = np.array(all_fix_on)[:n_train_tr]
d_on_train = np.array(all_d_on)[:n_train_tr]
test1_on_train = np.array(all_test1_on)[:n_train_tr]
# # test
# sp_test = np.concatenate(all_sp[n_train_tr:])
# print("Length of all trials concatenated: %s" % sp_test.shape)
# dm_test = np.concatenate(design_mat[n_train_tr:])
# print("Shape of dm: (%s, %s)" % dm_test.shape)
# len_tr_test = np.array(all_len_tr)[n_train_tr:]
# print("Number of trials for testing%s" % len_tr_test.shape)
# s_on_test = np.array(all_s_on)[n_train_tr:]
# fix_on_test = np.array(all_fix_on)[n_train_tr:]
# d_on_test = np.array(all_d_on)[n_train_tr:]
# test1_on_test = np.array(all_test1_on)[n_train_tr:]

x = dm_train
mu = np.mean(x, axis=0)
sigma = np.std(x, axis=0)
sigma0 = sigma
sigma0[sigma0 == 0] = 1
dm_train = (x - mu) / sigma0

T, _ = dm_train.shape
design_mat_offset = np.hstack((np.ones((T, 1)), dm_train))  # add a column of ones
lg_weights = (
    np.linalg.inv(design_mat_offset.T @ design_mat_offset)
    @ design_mat_offset.T
    @ sp_train
)
# sta=np.ones(sta.shape)
dt_fine = 1
# -- Make loss functions and minimize -----
loss_func = lambda prs: glm_functions.neg_log_lik_lnp(
    prs, design_mat_offset, sp_train, dt_fine, vals_to_return=0
)
grad_func = lambda prs: glm_functions.neg_log_lik_lnp(
    prs, design_mat_offset, sp_train, dt_fine, vals_to_return=1
)
hess_func = lambda prs: glm_functions.neg_log_lik_lnp(
    prs, design_mat_offset, sp_train, dt_fine, vals_to_return=2
)
optimizer = minimize(
    fun=loss_func,
    x0=lg_weights,
    jac=grad_func,
    hess=hess_func,
    method="trust-ncg",
    options={"disp": True, "gtol": 1e-6, "maxiter": 150},
)
filt_ML = optimizer.x
ntfilt = design_mat_offset.shape[1]
ttk = np.arange(-ntfilt + 1, 0) * dt_fine
fig = plt.figure(figsize=[12, 8])
plt.plot(ttk[10:30], ttk[10:30] * 0, c="k", linestyle="--")
plt.plot(ttk[10:30], filt_ML[10:30], c="darkorange", linewidth=4)
plt.xlabel("time before spike")
plt.ylabel("coefficient")
plt.title("Maximum likelihood filter estimate")
plt.tight_layout()
plt.savefig(outputpath + "hist_filt.svg", format="svg", bbox_inches="tight")


rate_pred_train = np.exp(filt_ML[0] + dm_train @ filt_ML[1:])

plt.subplots(figsize=(25, 10))
plt.plot(sp_train[20000:30000] / 10 - 0.1, "k")
avg_sp_train = firing_rate.moving_average(data=sp_train[20000:30000], win=20, step=1)
plt.plot(avg_sp_train)
plt.plot(rate_pred_train[20000:30000])
plt.tight_layout()
plt.savefig(outputpath + "pred.svg", format="svg", bbox_inches="tight")

# plot filters---------------------------------------
variables = list(indices.keys())
glm_vars = {}
glm_const = filt_ML[0:1]
glm_vars["glm_const"] = glm_const
for var in variables:
    st = indices[var][0]
    ed = indices[var][1]
    glm_vars[var] = filt_ML[st + 1 : ed + 1]

# print filters
variables = [
    "fix_cue",
    "fixation",
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
    "hist",
    "test_orient1",
    "test_orient2",
    "test_orient3",
    "test_orient4",
    "test_orient5",
    "test_orient6",
    "test_orient7",
    "test_orient8",
    "test_color1",
    "test_color2",
    "test_color3",
    "test_color4",
    "test_color5",
    "test_color6",
    "test_color7",
    "test_color8",
    "match",
    "bar_release",
]
cols = 6
rows = len(variables) // cols
rows = rows + 1 if len(variables) % cols > 0 else rows
f, ax = plt.subplots(rows, cols, figsize=(30, 20), sharey=True, sharex=True)
for iax, var in zip(np.concatenate(ax), variables):
    iax.plot(glm_vars[var])
    iax.set_title(var)

f.savefig(outputpath + "allfilters.svg", format="svg", bbox_inches="tight")


# componenets---------------------------
def get_component(var, dm, glm_vars, indices, start, end):
    st = indices[var][0]
    ed = indices[var][1]
    component = np.exp(dm[:, st:ed] @ glm_vars[var])
    return component[start:end]


cumsum = np.cumsum(len_tr_train)
cumsum = np.concatenate(([0], cumsum))
start = cumsum[:-1]
end = cumsum[1:]

all_components = {}
# fixation
var = "fix_cue"
# st = indices[var][0]
# ed = indices[var][1]
# component= np.exp( dm_train[:,st:ed] @ glm_vars[var])
# all_components[var] = component[start[0]:end[0]]
st = start[0]
ed = end[0]
all_components[var] = get_component(var, dm_train, glm_vars, indices, start=st, end=ed)
# sample
t_befs = 400
vars = ["s_o1", "s_o5", "s_c1", "s_c5", "s_n0"]
sampl_ids = [11, 55, 11, 55, 0]
for var, s_id in zip(vars, sampl_ids):
    idx = np.where(neu_data.sample_id[mask] == s_id)[0]
    st = int(start[idx[0]] + s_on_train[idx[0]] - t_befs)
    ed = end[idx[0]]
    all_components[var] = get_component(
        var, dm_train, glm_vars, indices, start=st, end=ed
    )
# delay
vars = ["d_o1", "d_o5", "d_c1", "d_c5", "d_n0"]
t_befd = 460 + 200 + 200
sampl_ids = [11, 55, 11, 55, 0]
for var, s_id in zip(vars, sampl_ids):
    idx = np.where(neu_data.sample_id[mask] == s_id)[0]
    st = int(start[idx[0]] + d_on_train[idx[0]] - t_befd)
    ed = end[idx[0]]
    all_components[var] = get_component(
        var, dm_train, glm_vars, indices, start=st, end=ed
    )
# test color
vars = [
    "test_color1",
    "test_color2",
    "test_color3",
    "test_color4",
    "test_color5",
    "test_color6",
    "test_color7",
    "test_color8",
]
t_befd = 450 + 460 + 200 + 200
sampl_ids = [1, 2, 3, 4, 5, 6, 7, 8]
for var, s_id in zip(vars, sampl_ids):
    idx, i_p = np.where(neu_data.test_stimuli[mask] % 10 == s_id)
    if idx[0] > n_train_tr:
        continue
    st = int(start[idx[0]] + test1_on_train[idx[0]] + 450 * i_p[0] - t_befd)
    ed = st + 600 + t_befd
    all_components[var] = get_component(
        var, dm_train, glm_vars, indices, start=st, end=ed
    )
# test orientation
vars = [
    "test_orient1",
    "test_orient2",
    "test_orient3",
    "test_orient4",
    "test_orient5",
    "test_orient6",
    "test_orient7",
    "test_orient8",
]
t_befd = 450 + 460 + 200 + 200
sampl_ids = [1, 2, 3, 4, 5, 6, 7, 8]
for var, s_id in zip(vars, sampl_ids):
    idx, i_p = np.where(neu_data.test_stimuli[mask] // 10 == s_id)
    if idx[0] > n_train_tr:
        continue
    st = int(start[idx[0]] + test1_on_train[idx[0]] + 450 * i_p[0] - t_befd)
    ed = st + 600 + t_befd
    all_components[var] = get_component(
        var, dm_train, glm_vars, indices, start=st, end=ed
    )

var = "fix_cue"
plt.subplots(figsize=(10, 5))
t = np.arange(len(all_components[var])) - s_on_train[0]
plt.plot(t, all_components[var], label=var)
plt.legend(
    fontsize=15, scatterpoints=5, columnspacing=0.5, framealpha=0, loc="upper right"
)
plt.suptitle("Component")
plt.savefig(outputpath + "fix_cue.svg", format="svg", bbox_inches="tight")

plt.subplots(figsize=(10, 5))
for var in ["s_o1", "s_o5", "s_c1", "s_c5", "s_n0"]:
    comp = all_components[var]
    t = np.arange(len(comp[200:-100])) - 200
    plt.plot(t, comp[200:-100], label=var)
plt.legend(
    fontsize=15, scatterpoints=5, columnspacing=0.5, framealpha=0, loc="upper right"
)
plt.suptitle("Component")
plt.savefig(outputpath + "sample.svg", format="svg", bbox_inches="tight")
#
plt.subplots(figsize=(10, 5))
for var in ["d_o1", "d_o5", "d_c1", "d_c5", "d_n0"]:
    comp = all_components[var]
    t = np.arange(len(comp[200:-100])) - 200
    plt.plot(t, comp[200:-100], label=var)
plt.legend(
    fontsize=15, scatterpoints=5, columnspacing=0.5, framealpha=0, loc="upper right"
)
plt.suptitle("Component")
plt.savefig(outputpath + "delay.svg", format="svg", bbox_inches="tight")
#
plt.subplots(figsize=(10, 5))
for var in [
    "test_color1",
    "test_color2",
    "test_color3",
    "test_color4",
    "test_color5",
    "test_color7",
    "test_color8",
]:  #'test_color6',
    comp = all_components[var]
    t = np.arange(len(comp[200:-100])) - 200
    plt.plot(t, comp[200:-100], label=var)
plt.legend(
    fontsize=15, scatterpoints=5, columnspacing=0.5, framealpha=0, loc="upper left"
)
plt.suptitle("Component")
plt.savefig(outputpath + "test_color.svg", format="svg", bbox_inches="tight")
#
plt.subplots(figsize=(10, 5))
for var in [
    "test_orient1",
    "test_orient2",
    "test_orient3",
    "test_orient4",
    "test_orient5",
    "test_orient7",
    "test_orient8",
]:  #'test_color6',
    comp = all_components[var]
    t = np.arange(len(comp[200:-100])) - 200
    plt.plot(t, comp[200:-100], label=var)
plt.legend(
    fontsize=15, scatterpoints=5, columnspacing=0.5, framealpha=0, loc="upper left"
)
plt.suptitle("Component")
plt.savefig(outputpath + "test_orient.svg", format="svg", bbox_inches="tight")
