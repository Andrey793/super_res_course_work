import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import cv2
import scipy


def synth_dataset(im, num_images, blur_sigma):
    """
    Creates a synthetic dataset to test the super-resolution (SR) algorithm.

    Parameters:
    im : ndarray
        Reference image (2D array).
    num_images : int
        Number of low-resolution images to generate.
    blur_sigma : float
        Standard deviation for Gaussian blur.

    Returns:
    images : list of ndarray
        List of randomly translated low-resolution images.
    offsets : ndarray
        Translation offsets for each low-resolution image.
    cropped_original : ndarray
        Cropped original image corresponding to the common area of low-res images.
    """
    pad_ratio = 0.2
    rows, cols = im.shape

    # Calculate the cropping indices
    working_row_sub = np.arange(
        max(0, int(0.5 * pad_ratio * rows) - 1), int((1 - 0.5 * pad_ratio) * rows + 1)
    )
    working_col_sub = np.arange(
        max(0, int(0.5 * pad_ratio * cols) - 1), int((1 - 0.5 * pad_ratio) * cols + 1)
    )

    # Crop the original image
    cropped_original = im[np.ix_(working_row_sub, working_col_sub)]

    # Initialize outputs
    images = []
    offsets = np.zeros((num_images, 2))

    # Add the first image (no translation)
    offsets[0, :] = [0, 0]
    images.append(cropped_original)
    #my_offsets = [0.7, -0.3, 0.2, -0.55]
    # Generate random translations for additional images
    for i in range(1, num_images):
        off = 2 * np.random.rand(1) - 1
        #off = [my_offsets[i]]
        offsets[i, :] = [off[0], off[0]]
        offset_row_sub = working_row_sub - offsets[i, 1]
        offset_col_sub = working_col_sub - offsets[i, 0]

        # Create a grid for interpolation
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        x2, y2 = np.meshgrid(offset_col_sub, offset_row_sub)

        # Interpolate the image at the translated positions
        translated_image = griddata(
            (x.ravel(), y.ravel()), im.ravel(), (x2, y2), method='linear', fill_value=0.5)
        images.append(translated_image)
    kernel = cv2.getGaussianKernel(ksize=3, sigma=blur_sigma)
    kernel = kernel @ kernel.T
    # Apply Gaussian blur to each image
    for i in range(num_images):
        #images[i] = gaussian_filter(images[i], sigma=blur_sigma)
        images[i] = scipy.ndimage.convolve(images[i], kernel)

        # Downsample the image by a factor of 2
        cur_im = images[i]
        images[i] = cur_im[1::2, 1::2]

    return images, offsets, cropped_original

