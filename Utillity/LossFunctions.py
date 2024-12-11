import numpy as np

def r2_score(y_true, y_pred, weights):
    """
    Calculate the sample weighted zero-mean R-squared score.

    Parameters:
    y_true (numpy.ndarray): Ground-truth values for responder_6.
    y_pred (numpy.ndarray): Predicted values for responder_6.
    weights (numpy.ndarray): Sample weight vector.

    Returns:
    float: The weighted zero-mean R-squared score.
    """
    numerator = np.sum(weights * (y_true - y_pred)**2)
    denominator = np.sum(weights * y_true**2)

    r2_score = 1 - numerator / denominator
    return r2_score