import torch
import torch.nn as nn
import numpy as np


# Model
class ExpModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim, bias=True)  # .float()#.float32()
        # print(self.linear.weight.dtype)
        # self.activation = nn.Softplus()

    def forward(self, x):
        out = torch.exp(self.linear(x))
        return out  # self.activation(out)


# Regularizers
class SmoothL2Regularization(nn.Module):
    def __init__(self, beta=0.01):
        super().__init__()
        self.beta = beta

    def forward(self, model):
        regularization_loss = 0.0
        for param in model.parameters():
            regularization_loss += torch.sum(torch.diff(param) ** 2)
        return 0.5 * self.beta * regularization_loss


class L1Regularization(nn.Module):
    def __init__(self, beta=0.01):
        super().__init__()
        self.beta = beta

    def forward(self, model):
        regularization_loss = 0.0
        for param in model.parameters():
            regularization_loss += torch.sum(torch.abs(param))
        return self.beta * regularization_loss


# Loss function
def poisson_glm_loss(y_hat, y):
    return -torch.mean(y * torch.log(y_hat) - y_hat)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            if (self.best_score - score) < self.delta:
                self.counter += 1
                self.save_checkpoint(val_loss, model)
                # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            self.best_score = score
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_idx_trials_train_val(idx_train, idx_val, idx_dict, seed):
    # Select trials for training and validation
    sn0 = idx_dict["sn0"][idx_train]
    s15 = idx_dict["s15"][idx_train]
    s11 = idx_dict["s11"][idx_train]
    s55 = idx_dict["s55"][idx_train]
    s51 = idx_dict["s51"][idx_train]
    idxs_train = np.concatenate((sn0, s15, s11, s55, s51))
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(idxs_train)
    # validation
    sn0 = idx_dict["sn0"][idx_val]
    s15 = idx_dict["s15"][idx_val]
    s11 = idx_dict["s11"][idx_val]
    s55 = idx_dict["s55"][idx_val]
    s51 = idx_dict["s51"][idx_val]
    idxs_val = np.concatenate((sn0, s15, s11, s55, s51))
    rng.shuffle(idxs_val)
    return idxs_train, idxs_val
