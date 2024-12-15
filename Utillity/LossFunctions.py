import numpy as np
import torch
from torch import nn


def r2_score(y_pred, y_true, weights):
    """
    Calculate the sample weighted zero-mean R-squared score.

    Parameters:
    y_true (numpy.ndarray): Ground-truth values for responder_6.
    y_pred (numpy.ndarray): Predicted values for responder_6.
    weights (numpy.ndarray): Sample weight vector.

    Returns:
    float: The weighted zero-mean R-squared score.
    """
    numerator = torch.sum(weights * (y_true - y_pred)**2)
    denominator = torch.sum(weights * y_true**2)

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