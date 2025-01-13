import numpy as np


def GaussianKernel(x, sigma):
    '''
    Args:
        x: array-like or float
        sigma: float - standard deviation
    Returns: numpy.ndarray or float = Gaussian kernel values
    '''
    return np.exp((-np.array(x) ** 2) / (2 * sigma ** 2))