from skimage import io
from rich import color
from scipy.optimize import root_scalar
import math
import numpy as np

def Get_Spatial_Res(image, sigma, K=3):
    """
    Args:
        image: 2d ndarray
        sigma: Gaussian blur sigma
        A: Brightness pulses amplitude
        K: some threshold, 2 < K < 5
        H(u, v): Frequency response function
        delta: discretization step
    Returns: spatial resolution l0 - float
    """
    delta = get_discrete_step(image)[0]
    #Brightness amplitude
    A = np.max(image) - np.min(image)
    H = frequency_response(
        sigma
    )
    #mean squared noise deviation
    sigma_v = get_noise_dev(image)
    Q = K * delta * sigma_v / (A * 2 * 5**0.5)
    def g(w, *args):
        return H(w, 0) - w*Q

    sol = root_scalar(g, args=(Q, ), bracket=(0, 1000), method='brentq')
    W = sol.root
    l0 = math.pi / W
    return l0

def frequency_response(sigma):
    """
    Args:
        sigma: gaussian blur sigma
    Returns: function
    """
    def H(u, v):
        return math.exp(-2 * (math.pi**2) * (sigma**2) * (u**2 + v**2))
    return H

def get_discrete_step(image):
    """
    Args:
        image: digital image
    Returns: (float, float)
    """
    height, width = image.shape

    # Compute the 2D Fourier Transform
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)  # Shift the zero frequency to the center
    magnitude_spectrum = np.abs(fft_shifted)

    # Compute spatial frequency axes
    freq_x = np.fft.fftfreq(width)
    freq_y = np.fft.fftfreq(height)
    freq_x = np.fft.fftshift(freq_x)
    freq_y = np.fft.fftshift(freq_y)

    # Estimate the highest significant frequency
    threshold = 0.01 * magnitude_spectrum.max()  # Consider frequencies above 1% of max magnitude
    high_freq_indices = np.where(magnitude_spectrum > threshold)
    max_freq_x = np.max(np.abs(freq_x[high_freq_indices[1]]))
    max_freq_y = np.max(np.abs(freq_y[high_freq_indices[0]]))

    # Estimate discretization step
    dx = 1 / (2 * max_freq_x)
    dy = 1 / (2 * max_freq_y)
    return (dx, dy)

def get_noise_dev(image):
    """
    Args:
        image: 2d ndarray
    Returns: float - mean squared noise deviation
    """
    # Compute Fourier Transform
    f_transform = np.fft.fft2(image)
    f_magnitude = np.abs(f_transform)

    # Focus on high frequencies (ignore low frequencies)
    high_freq_power = np.mean(f_magnitude[image.shape[0] // 4:, image.shape[1] // 4:])

    # Approximate MSND
    msnd = high_freq_power ** 2
    return msnd

im = io.imread('results/low_res_example.jpg')
if len(im.shape) == 3:
    im = color.rgb2gray(im)
im = np.resize(im, (205, 205))
im = im.astype(np.float64)
l0 = Get_Spatial_Res(im, 1)
print(l0, "low res resolution")

im = io.imread('results/super_v2.jpg')
if len(im.shape) == 3:
    im = color.rgb2gray(im)
im = np.resize(im, (512, 512))
im = im.astype(np.float64)
l0 = Get_Spatial_Res(im, 0.5)
print(l0, "super res resolution")

im = io.imread('results/reference.png')
if len(im.shape) == 3:
    im = color.rgb2gray(im)
im = np.resize(im, (512, 512))
im = im.astype(np.float64)
l0 = Get_Spatial_Res(im, 0)
print(l0, "reference resolution")
