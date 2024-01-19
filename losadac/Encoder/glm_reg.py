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
    "/session_struct/lip/neurons/2023-02-24_10-43-44_Riesling_lip_e1_r1_good1_neu.h5"
)
outputpath = "/envau/work/invibe/USERS/LOSADA/Users/losadac/results/"  # "/home/INT/losada.c/Documents/codes/results"
time_before = 1
select_block = 1
train_perc = 0.80
events = [
    "fix_cue",
    "hist",
    "s_o1",
    "s_o5",
    "s_c1",
    "s_c5",
    "s_n0",
    "s_on_off",
    "d_o1",
    "d_o5",
    "d_c1",
    "d_c5",
    "d_n0",
    "d_on_off",
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
fix_cue_dim = 10
sample_dim = 20
delay_dim = 20
test_dim = 20
match_dim = 20
bar_dim = 10
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
    btype="raised cosine", nBases=fix_cue_dim, duration=1200
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
    btype="raised cosine", nBases=bar_dim, duration=50
)
basis = {
    "fix_cue": fix_cue_basis,
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
        "sp": sp_sample_on[i_tr, 1 : len_tr + 1],
    }


n_tr, _ = sp_sample_on.shape
data = Parallel(n_jobs=-1)(delayed(get_dm)(i_tr) for i_tr in tqdm(range(n_tr)))

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
n_train_tr = int(n_tr * train_perc)
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
# test
sp_test = np.concatenate(all_sp[n_train_tr:])
print("Length of all trials concatenated: %s" % sp_test.shape)
dm_test = np.concatenate(design_mat[n_train_tr:])
print("Shape of dm: (%s, %s)" % dm_test.shape)
len_tr_test = np.array(all_len_tr)[n_train_tr:]
print("Number of trials for testing%s" % len_tr_test.shape)
s_on_test = np.array(all_s_on)[n_train_tr:]
fix_on_test = np.array(all_fix_on)[n_train_tr:]
d_on_test = np.array(all_d_on)[n_train_tr:]
test1_on_test = np.array(all_test1_on)[n_train_tr:]

## L2 regularization---------------------------------------------
logging.info("L2 reg start")
t_train, _ = dm_train.shape
t_test, _ = dm_test.shape
dm_train_offset = np.hstack((np.ones((t_train, 1)), dm_train))  # add a column of ones
dm_test_offset = np.hstack((np.ones((t_test, 1)), dm_test))
lg_weights = (
    np.linalg.inv(dm_train_offset.T @ dm_train_offset) @ dm_train_offset.T @ sp_train
)  # compute ML for initialization
# lg_weights = np.zeros(lg_weights.shape)
dt_fine = 1
lamvals = 2 ** np.arange(15)  # it's common to use a log-spaced set of values
nlam = len(lamvals)
ntfilt = dm_train_offset.shape[1]
# Precompute some quantities for training and test data
Imat = np.identity(ntfilt)  # identity matrix of size of filter + const
Imat[0, 0] = 0  # remove penalty on constant dc offset
# Allocate space for train and test errors
negLtrain = np.zeros(nlam)  # training error
negLtest = np.zeros(nlam)  # test error
w_ridge = np.zeros((ntfilt, nlam))  # filters for each lambda
optimizer_res = []
# Define train and test log-likelihood funcs
neglogli_train_func = lambda prs: glm_functions.neg_log_lik_lnp(
    prs, dm_train_offset, sp_train, dt_fine
)
# Now compute MAP estimate for each ridge parameter
wmap = lg_weights  # initialize parameter estimate
fig = plt.figure(figsize=[12, 8])
ttk = np.arange(-ntfilt + 1, 0) * dt_fine
plt.plot(ttk, ttk * 0, c="k", linestyle="--")
plt.ylabel("coefficient")
plt.xlabel("time before spike (s)")
for idx, lam in enumerate(lamvals):
    logging.info("lambda %d" % lam)
    # Compute ridge-penalized MAP estimate
    Cinv = lam * Imat  # set inverse prior covariance
    loss_post_func = lambda prs: glm_functions.neglogposterior(
        prs, neglogli_train_func, Cinv, vals_to_return=0
    )
    grad_post_func = lambda prs: glm_functions.neglogposterior(
        prs, neglogli_train_func, Cinv, vals_to_return=1
    )
    hess_post_func = lambda prs: glm_functions.neglogposterior(
        prs, neglogli_train_func, Cinv, vals_to_return=2
    )
    optimizer = minimize(
        fun=loss_post_func,
        x0=wmap,
        method="trust-ncg",
        jac=grad_post_func,
        hess=hess_post_func,
        tol=1e-6,
        options={"disp": False, "maxiter": 200},
    )
    wmap = optimizer.x
    print(optimizer.success)
    # Compute negative logli
    negLtrain[idx] = glm_functions.neg_log_lik_lnp(
        wmap, dm_train_offset, sp_train, dt_fine, vals_to_return=0
    )
    negLtest[idx] = glm_functions.neg_log_lik_lnp(
        wmap, dm_test_offset, sp_test, dt_fine, vals_to_return=0
    )  # test loss
    print(negLtrain[idx])
    # store the filter
    w_ridge[:, idx] = wmap

    # plot it
    plt.plot(ttk, wmap[1:], linewidth=2, label="lambda: " + str(lam))
plt.legend()
plt.tight_layout()
plt.savefig(outputpath + "l2reg.svg", format="svg", bbox_inches="tight")

imin = np.argmin(negLtest)
fig = plt.figure(figsize=[12, 8])
plt.subplot(222)
plt.plot(ttk, w_ridge[1:, :])
plt.title("all ridge estimates")
plt.subplot(221)
plt.semilogx(lamvals, negLtrain, "o-", linewidth=4)
plt.vlines(lamvals[imin], np.min(negLtrain), np.max(negLtrain), "k", linestyles=":")
plt.title("training logli")
plt.subplot(223)
plt.semilogx(lamvals, negLtest, "-o", linewidth=4)
plt.vlines(lamvals[imin], np.min(negLtest), np.max(negLtest), "k", linestyles=":")
plt.xlabel("lambda")
plt.title("test logli")
filt_ridge = w_ridge[1:, imin]
plt.subplot(224)
# plt.plot(ttk, ttk*0, c='k', linestyle='--')
plt.plot(ttk[:20], filt_ridge[10:30], linewidth=4)
plt.xlabel("time before spike (s)")
plt.title("best ridge estimate (lamda: %d)" % lamvals[imin])
plt.tight_layout()
plt.savefig(outputpath + "l2reg_train_test.svg", format="svg", bbox_inches="tight")
logging.info("L2 Smoothing start")
## L2 Smoothing--------------------------------------------------------------
t_train, _ = dm_train.shape
t_test, _ = dm_test.shape
dm_train_offset = np.hstack((np.ones((t_train, 1)), dm_train))  # add a column of ones
dm_test_offset = np.hstack((np.ones((t_test, 1)), dm_test))
ntfilt = dm_train_offset.shape[1]
# This matrix computes differences between adjacent coeffs
Dx1 = diags(
    (np.ones((ntfilt - 1, 1)) @ [[1, -1]]).T, np.arange(2), (ntfilt, ntfilt - 1)
).A
Dx = Dx1.T @ Dx1  # computes squared diffs
# Embed Dx matrix in matrix with one extra row/column for constant coeff
D = block_diag(0, Dx)
dt_fine = 1
lamvals_sm = 2 ** np.arange(15)  # it's common to use a log-spaced set of values
nlam = len(lamvals_sm)
negLtrain_sm = np.zeros(nlam)  # training error
negLtest_sm = np.zeros(nlam)  # test error
w_smooth = np.zeros((ntfilt, nlam))  # filters for each lambda
optimizer_res = []
# Define train and test log-likelihood funcs
neglogli_train_func = lambda prs: glm_functions.neg_log_lik_lnp(
    prs, dm_train_offset, sp_train, dt_fine
)
# Now compute MAP estimate for each ridge parameter
wmap = lg_weights  # np.zeros(ntfilt)  #initialize parameter estimate
fig = plt.figure(figsize=[12, 8])
ttk = np.arange(-ntfilt + 1, 0) * dt_fine
plt.plot(ttk, ttk * 0, c="k", linestyle="--")
plt.ylabel("coefficient")
plt.xlabel("time before spike (s)")
for idx, lam in enumerate(lamvals_sm):
    logging.info("lambda %d" % lam)
    # Compute ridge-penalized MAP estimate
    Cinv = lam * D  # set inverse prior covariance
    loss_post_func = lambda prs: glm_functions.neglogposterior(
        prs, neglogli_train_func, Cinv, vals_to_return=0
    )
    grad_post_func = lambda prs: glm_functions.neglogposterior(
        prs, neglogli_train_func, Cinv, vals_to_return=1
    )
    hess_post_func = lambda prs: glm_functions.neglogposterior(
        prs, neglogli_train_func, Cinv, vals_to_return=2
    )
    optimizer = minimize(
        fun=loss_post_func,
        x0=wmap,
        method="trust-ncg",
        jac=grad_post_func,
        hess=hess_post_func,
        tol=1e-6,
        options={"disp": False, "maxiter": 200},
    )
    wmap = optimizer.x
    print(optimizer.success)
    # Compute negative logli
    negLtrain_sm[idx] = glm_functions.neg_log_lik_lnp(
        wmap, dm_train_offset, sp_train, dt_fine, vals_to_return=0
    )
    negLtest_sm[idx] = glm_functions.neg_log_lik_lnp(
        wmap, dm_test_offset, sp_test, dt_fine, vals_to_return=0
    )  # test loss
    print(negLtrain_sm[idx])
    # store the filter
    w_smooth[:, idx] = wmap

    # plot it
    plt.plot(ttk, wmap[1:], linewidth=2, label="lambda: " + str(lam))
plt.legend()
plt.tight_layout()
plt.savefig(outputpath + "l2smooth.svg", format="svg", bbox_inches="tight")

### ===== Plot filter estimates and errors for ridge estimates =====
imin = np.argmin(negLtest_sm)
fig = plt.figure(figsize=[12, 8])
plt.subplot(222)
plt.plot(ttk, w_smooth[1:, :])
plt.title("all ridge estimates")
plt.subplot(221)
plt.semilogx(lamvals_sm, negLtrain_sm, "o-", linewidth=4)
plt.vlines(
    lamvals_sm[imin], np.min(negLtrain_sm), np.max(negLtrain_sm), "k", linestyles=":"
)
plt.title("training logli")
plt.subplot(223)
plt.semilogx(lamvals_sm, negLtest_sm, "-o", linewidth=4)
plt.vlines(
    lamvals_sm[imin], np.min(negLtest_sm), np.max(negLtest_sm), "k", linestyles=":"
)
plt.xlabel("lambda")
plt.title("test logli")
filt_smooth = w_smooth[1:, imin]
plt.subplot(224)
# plt.plot(ttk, ttk*0, c='k', linestyle='--')
plt.plot(ttk[:20], filt_smooth[10:30], linewidth=4)
plt.xlabel("time before spike (s)")
plt.title("best ridge estimate (lamda: %d)" % lamvals_sm[imin])

plt.tight_layout()
plt.savefig(outputpath + "l2smooth_train_test.svg", format="svg", bbox_inches="tight")
