import numpy as np

def linear_kernel(x):
    """
    Args:
        x: np.ndarray

    Returns: np.ndarray
    """
    y = np.zeros_like(x)
    above_thresh = np.abs(x) >= 1
    below_thresh = ~above_thresh

    y[below_thresh] = 1 - np.abs(x[below_thresh])
    return y
