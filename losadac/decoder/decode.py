import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from itertools import permutations
from tqdm import tqdm

seed = 2023
np.random.seed(seed)


def reorganize_data(data, num_iter):
    """Reorganizes the input data into structured arrays."""
    conditions = [
        "sample",
        "orientation",
        "color",
        "neutral",
        "sample nm1",
        "orientation nm1",
        "color nm1",
        "neutral nm1",
    ]
    ps_a = {
        cond: np.full(
            (num_iter, data[0][cond].shape[0], data[0][cond].shape[0]), np.nan
        )
        for cond in conditions
    }

    for i in range(num_iter):
        for cond in conditions:
            ps_a[cond][i, :] = data[i][cond]

    return {"perf sample": ps_a}


def pick_train_test_trials(trials, train_ratio):
    """Splits trials into training and testing sets."""
    shuffled_trials = np.random.permutation(trials)
    split_idx = int(len(trials) * train_ratio)
    train_trials = shuffled_trials[:split_idx]
    test_trials = shuffled_trials[split_idx:]

    if np.any(np.in1d(train_trials, test_trials)):
        print("Warning: Overlapping trials in training and testing sets.")

    return train_trials, test_trials


def prepare_data_sets(datas, trial_indices, num_train, num_test):
    """Prepare training and testing datasets."""
    num_cells = len(datas)
    data_shape = datas[0].shape[1]

    train_data = np.empty((data_shape, num_train * 2, num_cells))
    test_data = np.empty((data_shape, num_test * 2, num_cells))

    for i in range(num_cells):
        trials_train, trials_test = pick_train_test_trials(
            trial_indices[i], 1 - num_test / num_train
        )
        train_data[:, :num_train, i] = datas[i][
            :, np.random.choice(trials_train, num_train)
        ]
        train_data[:, num_train:, i] = datas[i][
            :, np.random.choice(trials_train, num_train)
        ]
        test_data[:, :num_test, i] = datas[i][
            :, np.random.choice(trials_test, num_test)
        ]
        test_data[:, num_test:, i] = datas[i][
            :, np.random.choice(trials_test, num_test)
        ]

    return train_data, test_data


def decode_conditions(d):
    """Performs decoding analysis on the data using an SVM classifier."""
    dat = d["data"]
    num_cells = d["num selected lip"]

    num_train = 30
    num_test = 10

    selected_cells = np.random.choice(len(dat), num_cells, replace=False)

    names = [dat[i]["name"] for i in selected_cells]
    datas = [dat[i]["Sample zscored"] for i in selected_cells]
    datat = [dat[i]["Test1 zscored"] for i in selected_cells]
    sample_ids = [dat[i]["Sample Id"] for i in selected_cells]
    test_ids = [dat[i]["Test Id"] for i in selected_cells]
    positions = [dat[i]["position"] for i in selected_cells]

    # Define trial conditions
    trial_conditions = {
        "o1c1": [(sid == 11) & (pos == 1) for sid, pos in zip(sample_ids, positions)],
        "o1c5": [(sid == 15) & (pos == 1) for sid, pos in zip(sample_ids, positions)],
        "o5c1": [(sid == 51) & (pos == 1) for sid, pos in zip(sample_ids, positions)],
        "o5c5": [(sid == 55) & (pos == 1) for sid, pos in zip(sample_ids, positions)],
        "o1": [
            (sid // 10 == 1) & (pos == 1) for sid, pos in zip(sample_ids, positions)
        ],
        "o5": [
            (sid // 10 == 5) & (pos == 1) for sid, pos in zip(sample_ids, positions)
        ],
        "c1": [(sid % 10 == 1) & (pos == 1) for sid, pos in zip(sample_ids, positions)],
        "c5": [(sid % 10 == 5) & (pos == 1) for sid, pos in zip(sample_ids, positions)],
        "n": [(sid == 0) & (pos == 1) for sid, pos in zip(sample_ids, positions)],
        "nn": [(sid != 0) & (pos == 1) for sid, pos in zip(sample_ids, positions)],
    }

    # Prepare training and test data sets
    train_data_c, test_data_c = prepare_data_sets(
        datas, trial_conditions["c1"], num_train, num_test
    )
    train_data_o, test_data_o = prepare_data_sets(
        datas, trial_conditions["o1"], num_train, num_test
    )
    train_data_n, test_data_n = prepare_data_sets(
        datas, trial_conditions["n"], num_train, num_test
    )
    train_data_s, test_data_s = prepare_data_sets(
        datas, trial_conditions["o1c1"], num_train, num_test
    )

    # Training labels
    y_train_c = np.concatenate([np.zeros(num_train), np.ones(num_train)])
    y_test_c = np.concatenate([np.zeros(num_test), np.ones(num_test)])
    y_train_s = np.tile([0, 1, 2, 3], num_train)
    y_test_s = np.tile([0, 1, 2, 3], num_test)

    # SVM training and performance evaluation
    clf = SVC(kernel="linear", random_state=seed)

    # Example: Training on color trials
    clf.fit(train_data_c.T, y_train_c)
    perf_c = clf.score(test_data_c.T, y_test_c)

    # You can add other training/testing conditions as needed

    return perf_c


# Example usage
d = {
    "data": [],  # Populate this with actual data
    "num selected lip": 10,
}
performance = decode_conditions(d)
print(f"Decoding performance: {performance}")
