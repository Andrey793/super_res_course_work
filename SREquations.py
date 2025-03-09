import numpy as np
from scipy.sparse import coo_matrix, vstack

from sptoeplitz import sptoeplitz
from linear_kernel import linear_kernel
from GaussianKernel import GaussianKernel


def sr_equations(images, offsets, blur_sigma):
    """
    Creates the Super-Resolution linear equations for the given data.

    Parameters:
        images (list of numpy.ndarray): List of low-resolution images.
        offsets (numpy.ndarray): Array of offsets for each image.
        blur_sigma (float): Standard deviation of the Gaussian blur.

    Returns:
        tuple: lhs (scipy.sparse matrix), rhs (numpy.ndarray)
    """
    lhs = []
    rhs = []
    scale = 2
    super_size = scale * np.array(images[0].shape) + [1, 1]

    for i in range(len(images)):
        trans_mat = trans_mat_func(super_size, offsets[i])
        blur_mat1 = blur_mat(super_size, blur_sigma)
        dec_mat = dec_mat_func(super_size, scale)
        cur_lhs = dec_mat @ blur_mat1 @ trans_mat
        cur_rhs = images[i].flatten()

        lhs.append(cur_lhs)
        rhs.append(cur_rhs)

    lhs = vstack(lhs)
    rhs = np.concatenate(rhs)
    return lhs, rhs


def trans_mat_func(super_size, offsets):
    """
    Creates a translation operator.
    """
    transpose_mat = transpose_mat_func(super_size)
    return (
            transpose_mat @ trans_mat_y(super_size, offsets[0])
            @ transpose_mat @ trans_mat_y(super_size, offsets[1])
    )


def trans_mat_y(super_size, offset):
    """
    Creates a translation operator for the Y axis.
    """
    n = np.prod(super_size)
    row1 = np.zeros(n)
    nz_ind = np.arange(np.floor(1 - offset), np.ceil(1 - offset) + 1, dtype=int)
    filter_values = linear_kernel(1 - offset - nz_ind)
    nz_ind = np.array(nz_ind) - 1
    nz_ind[nz_ind < 0] += n
    nz_ind[nz_ind >= n] -= n
    row1[nz_ind] = filter_values

    col1 = np.zeros(n)
    col1[0] = row1[0]
    col1[1] = row1[-1]
    return sptoeplitz(col1, row1)


def transpose_mat_func(super_size):
    """
    Creates a matrix transposition operator.
    """
    input_pix_ind = np.arange(np.prod(super_size)).reshape(super_size[0], super_size[1])
    output_pix_ind = np.arange(np.prod(super_size)).reshape(super_size[1], super_size[0]).T
    result = coo_matrix((np.ones_like(input_pix_ind).flatten(), (output_pix_ind.flatten(), input_pix_ind.flatten())))
    return result


def blur_mat(super_size, blur_sigma):
    """
    Creates a blurring operator.
    """
    transpose_mat = transpose_mat_func(super_size)
    return (
            transpose_mat @ blur_mat_y(super_size, blur_sigma)
            @ transpose_mat @ blur_mat_y(super_size, blur_sigma)
    )


def blur_mat_y(super_size, blur_sigma):
    """
    Creates a blurring operator for the Y axis.
    """
    blur_kernel = GaussianKernel(np.arange(-2, 3), blur_sigma)
    blur_kernel /= np.sum(blur_kernel)

    n = np.prod(super_size)
    row1 = np.zeros(n)
    row1[[-2, -1, 0, 1, 2]] = blur_kernel

    col1 = np.zeros(n)
    col1[0] = row1[0]
    col1[1] = row1[1]

    return sptoeplitz(col1, row1)


def dec_mat_func(super_size, scale):
    """
    Creates a decimation operator.
    """
    sampled_size = (np.array(super_size) - 1) // scale
    output_row, output_col = np.meshgrid(range(sampled_size[0]), range(sampled_size[1]))
    input_row = scale * (output_row + 1)
    input_col = scale * (output_col + 1)
    input_ind = np.ravel_multi_index((input_col - 1, input_row - 1), super_size)
    output_ind = np.ravel_multi_index((output_col, output_row), sampled_size)
    res = coo_matrix((np.ones_like(output_ind).flatten(), (output_ind.flatten(), input_ind.flatten())),
                      shape=(np.prod(sampled_size), np.prod(super_size)))
    return res
