import numpy as np
from ephysvibe.task import task_constants
from scipy import signal
import glm_functions


## Basis ----------------------------------------------------------------------
def compute_smooth_temporal_basis(btype, nBases, duration):
    nkbins = duration
    ttb = np.array(
        [np.arange(1, nkbins + 1).tolist()] * nBases
    ).T  # time indices for basis
    if btype == "raised cosine":
        # spacing between the centers must be 1/4 of the with of the cosine
        dbcenter = nkbins / (3 + nBases)  # spacing between bumps
        width = 4 * dbcenter  # width of each bump
        bcenters = 2 * dbcenter + dbcenter * np.arange(0, nBases)
        x = ttb - np.array([bcenters.tolist()] * nkbins)
        period = width
        basis = (abs(x / period) < 0.5) * (np.cos(x * 2 * np.pi / period) * 0.5 + 0.5)
    elif btype == "boxcar":
        width = nkbins / nBases
        basis = np.zeros(np.shape(ttb))
        bcenters = width * np.arange(1, nBases + 1) - width / 2
        for k in np.arange(0, nBases):
            idx = np.logical_and(
                ttb[:, k] > np.round(width * k), ttb[:, k] <= np.round(width * (k + 1))
            )
            basis[idx, k] = 1 / sum(idx)

    return basis


def make_non_linear_raised_cos(nBases, binSize, endPoints, nlOffset=1):
    # Make nonlinearly stretched basis consisting of raised cosines.
    # Nonlinear stretching allows faster changes near the event.
    #
    # 	nBases: [1] - # of basis vectors
    # 	binSize: time bin size (separation for representing basis
    #   endPoints: [2 x 1] = 2-vector containg [1st_peak  last_peak], the peak
    #          (i.e. center) of the last raised cosine basis vectors (in ms)
    #   nlOffset: [1] offset for nonlinear stretching of x axis (in ms):
    #          y = log(t+nlOffset) (larger nlOffset -> more nearly linear stretching)
    #
    #  Outputs:  iht = time lattice on which basis is defined
    #            ihbasis = basis itself
    #            ihctrs  = centers of each basis function
    #

    # nonlinearity for stretching x axis (and its inverse)
    def nlin(x):
        return np.log(x + 1e-20)

    def invnl(x):
        return np.exp(x - 1e-20)

    if nlOffset <= 0:
        print("nlOffset must be greater than 0")

    yrnge = nlin(endPoints + nlOffset)
    db = (np.diff(yrnge) / (nBases - 1))[0]  # spacing between raised cosine peaks
    ctrs = np.arange(yrnge[0], yrnge[1] + db, db)  # centers for basis vectors
    mxt = invnl(yrnge[1] + 2 * db) - nlOffset  # maximum time bin
    iht = np.arange(0, mxt, binSize) / binSize

    def ff(x, c, dc):
        return (
            np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x - c) * np.pi / dc / 2))) + 1
        ) / 2

    x_1 = nlin(iht + nlOffset)
    x = np.array([x_1.tolist()] * nBases).T
    c = np.array([ctrs.tolist()] * len(iht))
    ihbasis = ff(x, c, db)
    hctrs = invnl(ctrs)

    return ihbasis


# Task variables/stimuli --------------------------------------------------------
def def_sample_stim(stim, len_tr, e_time, sample_id):
    # sample on off
    stim["s_on_off"] = np.zeros(len_tr)
    stim["s_on_off"][int(e_time[4]) : int(e_time[5])] = 1  # : int(e_time[5])
    # sample features
    stim["s_11"] = np.zeros(len_tr)
    stim["s_15"] = np.zeros(len_tr)
    stim["s_55"] = np.zeros(len_tr)
    stim["s_51"] = np.zeros(len_tr)
    stim["s_n0"] = np.zeros(len_tr)

    if sample_id == 11:
        stim["s_11"][int(e_time[4]) : int(e_time[5])] = 1
    elif sample_id == 15:
        stim["s_15"][int(e_time[4]) : int(e_time[5])] = 1
    elif sample_id == 0:
        stim["s_n0"][int(e_time[4]) : int(e_time[5])] = 1
    elif sample_id == 55:
        stim["s_55"][int(e_time[4]) : int(e_time[5])] = 1
    elif sample_id == 51:
        stim["s_51"][int(e_time[4]) : int(e_time[5])] = 1
    return stim


def def_delay_stim(stim, len_tr, e_time, sample_id):
    # delay on off
    stim["d_on_off"] = np.zeros(len_tr)
    stim["d_on_off"][int(e_time[5]) : int(e_time[6])] = 1  #: int(e_time[6])
    stim["d_11"] = np.zeros(len_tr)
    stim["d_15"] = np.zeros(len_tr)
    stim["d_55"] = np.zeros(len_tr)
    stim["d_51"] = np.zeros(len_tr)
    stim["d_n0"] = np.zeros(len_tr)

    if sample_id == 11:
        stim["d_11"][int(e_time[5]) : int(e_time[6])] = 1
    elif sample_id == 15:
        stim["d_15"][int(e_time[5]) : int(e_time[6])] = 1
    elif sample_id == 0:
        stim["d_n0"][int(e_time[5]) : int(e_time[6])] = 1
    elif sample_id == 55:
        stim["d_55"][int(e_time[5]) : int(e_time[6])] = 1
    elif sample_id == 51:
        stim["d_51"][int(e_time[5]) : int(e_time[6])] = 1
    return stim


def def_stim(events, len_tr, e_time, e_code, sample_id, sp, test_stimuli):
    stim = {}
    # define stimuli
    if "hist" in events:
        stim["hist"] = np.concatenate(([0], sp))
    # fix_cue
    if "fix_cue" in events:
        stim["fix_cue"] = np.zeros(len_tr)
        stim["fix_cue"][int(e_time[2])] = 1
        # fixation
    if "fixation" in events:
        t_off_code = task_constants.EVENTS_B1["fix_spot_off"]
        idx_off = np.where(e_code == t_off_code)[0]
        if len(idx_off) == 0:
            idx_off = -1
        else:
            idx_off = idx_off[0]
        stim["fixation"] = np.zeros(len_tr)
        stim["fixation"][int(e_time[3]) : int(e_time[idx_off])] = 1
    # samples
    if np.any(np.isin(["s_11", "s_15", "s_51", "s_55", "s_n0"], events)):
        stim = def_sample_stim(stim, len_tr, e_time, sample_id)
    # delay
    if np.any(np.isin(["d_11", "d_15", "d_51", "d_55", "d_n0"], events)):
        stim = def_delay_stim(stim, len_tr, e_time, sample_id)
    # test stimuli on off
    if np.any(
        np.isin(
            [
                "match",
                "test_on_off",
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
            ],
            events,
        )
    ):
        stim["match"] = np.zeros(len_tr)
        n_test = np.sum(~np.isnan(test_stimuli))
        max_n_test = test_stimuli.shape[0]
        for i_t in np.arange(n_test) + 1:
            stim["test_on_off" + str(i_t)] = np.zeros(len_tr)
            t_on_code = task_constants.EVENTS_B1["test_on_" + str(i_t)]
            t_off_code = task_constants.EVENTS_B1["test_off_" + str(i_t)]
            idx_on = np.where(e_code == t_on_code)[0]
            idx_off = np.where(e_code == t_off_code)[0]
            if i_t <= n_test:
                if i_t == 4 and len(idx_off) == 0:
                    idx_off = np.where(
                        e_code == task_constants.EVENTS_B1["test_off_5"]
                    )[0]
                if i_t == n_test and sample_id == test_stimuli[n_test - 1]:
                    stim["match"][int(e_time[idx_on])] = 1
                stim["test_on_off" + str(i_t)][
                    int(e_time[idx_on])
                ] = 1  # : int(e_time[idx_off])
        # test stimuli
        for c_o in np.arange(1, 9):
            # orientation
            idx_stim = np.where(test_stimuli // 10 == c_o)[0]
            codes = idx_stim * 2 + 6
            stim["test_orient" + str(c_o)] = np.zeros(len_tr)
            if len(codes) != 0:
                for i_c, i_s in zip(codes, idx_stim):
                    t_off_code = task_constants.EVENTS_B1["test_off_" + str(i_s + 1)]
                    idx_off = np.where(e_code == t_off_code)[0]
                    if len(idx_off) == 0 and i_s + 1 == 4:
                        t_off_code = task_constants.EVENTS_B1["test_off_5"]
                        idx_off = np.where(e_code == t_off_code)[0]
                    stim["test_orient" + str(c_o)][
                        int(e_time[i_c]) : int(e_time[idx_off[0]])
                    ] = 1  #:int(e_time[i_c+1])
            # color
            idx_stim = np.where(test_stimuli % 10 == c_o)[0]
            codes = idx_stim * 2 + 6
            stim["test_color" + str(c_o)] = np.zeros(len_tr)
            if len(codes) != 0:
                for i_c, i_s in zip(codes, idx_stim):
                    t_off_code = task_constants.EVENTS_B1["test_off_" + str(i_s + 1)]
                    idx_off = np.where(e_code == t_off_code)[0]
                    if len(idx_off) == 0 and i_s + 1 == 4:
                        t_off_code = task_constants.EVENTS_B1["test_off_5"]
                        idx_off = np.where(e_code == t_off_code)[0]
                    stim["test_color" + str(c_o)][
                        int(e_time[i_c]) : int(e_time[idx_off[0]])
                    ] = 1  #:int(e_time[i_c+1])
        # bar_release
        stim["bar_release"] = np.zeros(len_tr)
        bar_release = np.where(e_code == 4)[0]
        if len(bar_release) == 1:
            stim["bar_release"][int(e_time[bar_release])] = 1

    return stim


# Regularization ------------------------------------------------


def neg_log_lik_lnp(thetas, xx, yy, dt_bin, vals_to_return=3):
    """Compute negative log-likelihood of data under Poisson GLM model with
        exponential nonlinearity.

        Args
        ----
        thetas: ndarray (d X 1)
            parameter vector
        xx: ndarray (T X d)
            design matrix
        yy: ndarray (T X 1)
            response variable (spike count per time bin)
        dt_bin: float
            time bin size used
        vals_to_return: int
            which of negative log-likelihood (0), gradient (1), or hessian (2) to return.
            (3) returns all three values. This is necessary due to scipy.optimize.minimize
            requiring the three separate functions with a single return value for each    stim['s_o1']= np.zeros(len_tr)
    stim['s_o5']= np.zeros(len_tr)
    stim['s_c1']= np.zeros(len_tr)
    stim['s_c5']= np.zeros(len_tr).

        Returns
        -------
        neglogli: float
            negative log likelihood of spike train
        dL: ndarray (d X 1)
            gradient
        H: ndarray (d X d)
            Hessian (second derivative matrix)
    """
    thetas = np.array(thetas, dtype=np.float128)
    xx = np.array(xx, dtype=np.float128)
    yy = np.array(yy, dtype=np.float128)

    # Compute GLM filter output and conditional intensity
    xx_theta = xx @ thetas  # filter output
    f_rate = np.exp(xx_theta) * dt_bin  # conditional intensity (per bin)

    # ---------  Compute log-likelihood objective function -----------
    Trm1 = -xx_theta.T @ yy  #  spike term from Poisson log-likelihood
    Trm0 = np.sum(f_rate)  # non-spike term
    neglogli = Trm1 + Trm0

    # ---------  Compute Gradient -----------------
    dL1 = -xx.T @ yy  # spiking term (the spike-triggered average)
    dL0 = xx.T @ f_rate  # non-spiking term
    dL = dL1 + dL0

    # ---------  Compute Hessian -------------------
    H = xx.T @ (xx * np.transpose([f_rate]))  # non-spiking term

    # neglogli, dL, H=np.array(neglogli,dtype=np.float128), np.array(dL,dtype=np.float128), np.array(H ,dtype=np.float128)
    if vals_to_return == 3:
        return neglogli, dL, H
    else:
        return [neglogli, dL, H][vals_to_return]


def neglogposterior(thetas, neglogli_fun, Cinv, vals_to_return=3):
    """Compute negative log-posterior given a negative log-likelihood function
    and zero-mean Gaussian prior with inverse covariance 'Cinv'.

    # Compute negative log-posterior by adding quadratic penalty to log-likelihood

    Args
    ----
    thetas: ndarray (d X 1)
        parameter vector
    neglogli_fun: callable
        function that computes negative log-likelihood, gradient, and hessian.
    Cinv: ndarray (d X d)
        inverse covariance of prior
    vals_to_return: int
        which of negative log-posterior (0), gradient (1), or hessian (2) to return.
        (3) returns all three values. This is necessary due to scipy.optimize.minimize
        requiring the three separate functions with a single return value for each.

    Returns
    -------
    neglogpost: float
        negative log posterior
    grad: ndarray (d X 1)
        gradient
    H: ndarray (d X d)
        Hessian (second derivative matrix)
    """

    neglogpost, grad, H = neglogli_fun(thetas)
    neglogpost = neglogpost + 0.5 * thetas.T @ Cinv @ thetas
    grad = grad + Cinv @ thetas
    H = H + Cinv

    if vals_to_return == 3:
        return neglogpost, grad, H
    else:
        return [neglogpost, grad, H][vals_to_return]


def get_dm(i_tr, last_event, events, neu_data, time_before, sp_sample_on, mask, basis):
    print(i_tr)
    print(neu_data.sample_id[mask][i_tr])
    e_time = (
        neu_data.code_samples[mask][i_tr]
        - neu_data.code_samples[mask][i_tr][4]
        + time_before
    )
    e_code = neu_data.code_numbers[mask][i_tr]
    test_stimuli = neu_data.test_stimuli[mask][i_tr]
    # n_test=np.sum(~np.isnan(test_stimuli))
    reward = task_constants.EVENTS_B1[last_event]
    idx_last = np.where(e_code == reward)[0]

    len_tr = int(e_time[idx_last][0] + 200)
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
            sb_conv = signal.convolve2d(stim[key][:, np.newaxis], base)
            all_dm[key] = sb_conv[:len_tr]
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
        "sample_id": sample_id,
    }


def get_dm_mix(
    i_tr, last_event, events, neu_data, time_before, sp_sample_on, mask, basis
):
    e_time = (
        neu_data.code_samples[mask][i_tr]
        - neu_data.code_samples[mask][i_tr][4]
        + time_before
    )
    e_code = neu_data.code_numbers[mask][i_tr]
    test_stimuli = neu_data.test_stimuli[mask][i_tr]
    # n_test=np.sum(~np.isnan(test_stimuli))
    reward = task_constants.EVENTS_B1[last_event]
    idx_last = np.where(e_code == reward)[0]

    len_tr = int(e_time[idx_last][0] + 200)
    rng = np.random.default_rng(seed=2024)
    a = neu_data.sample_id[mask].copy()
    rng.shuffle(a)
    sample_id = a[i_tr]
    # print(a[i_tr], neu_data.sample_id[mask][i_tr])
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
            sb_conv = signal.convolve2d(stim[key][:, np.newaxis], base)
            all_dm[key] = sb_conv[:len_tr]
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
