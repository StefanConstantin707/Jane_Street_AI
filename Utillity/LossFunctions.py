import numpy as np
import torch
from torch import nn


def r2_score(y_pred, y_true, weights):
    numerator = torch.sum(weights * (y_true - y_pred)**2)
    denominator = torch.sum(weights * y_true**2)

    r2_score = 1 - numerator / denominator
    return r2_score


def r2_score_numpy(y_pred, y_true, weights):
    numerator = np.sum(weights * (y_true - y_pred)**2)
    denominator = np.sum(weights * y_true**2)

    r2_score = 1 - numerator / denominator
    return r2_score


def r2_score_batch(y_pred, y_true, weights):
    numerator = np.sum(weights * (y_true - y_pred)**2, axis=1)
    denominator = np.sum(weights * y_true**2, axis=1)

    r2_score = 1 - numerator / denominator
    return r2_score


def r2_loss(y_pred, y_true, weights):
    numerator = torch.sum(weights * (y_true - y_pred)**2)
    denominator = torch.sum(weights * y_true**2)

    r2_loss = numerator / denominator
    return r2_loss


def weighted_mse(y_pred, y_true, weights):
    loss_fct = nn.MSELoss(reduction='none')

    unweighted_loss = loss_fct(y_pred, y_true)
    weighted_loss = weights * unweighted_loss

    return weighted_loss.mean()


def weighted_mse_r6(y_pred, y_true, weights):
    loss_fct = nn.MSELoss(reduction='none')

    y_pred = y_pred[:, 6]
    y_true = y_true[:, 6]
    weights = weights.squeeze()

    unweighted_loss = loss_fct(y_pred, y_true)
    weighted_loss = weights * unweighted_loss

    return weighted_loss.mean()


def weighted_mse_r6_weighted(y_pred, y_true, weights):
    loss_fct = nn.MSELoss(reduction='none')
    loss_weights = torch.tensor([1, 1, 1, 1, 1, 1, 2, 1, 1], dtype=torch.float32, device=y_pred.device)
    loss_weights = loss_weights / loss_weights.sum()

    unweighted_loss = loss_fct(y_pred, y_true)
    weighted_loss = weights * unweighted_loss

    double_weighted_loss = loss_weights * weighted_loss

    return double_weighted_loss.mean()