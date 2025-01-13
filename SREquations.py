import numpy as np
from scipy.sparse import coo_matrix, vstack

from IBP.pythonProject.sptoeplitz import sptoeplitz
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
    super_size = 2 * np.array(images[0].shape) + [1, 1]
    #print('offsets')
    #print(offsets)

    for i in range(len(images)):
        trans_mat = trans_mat_func(super_size, offsets[i])
        blur_mat1 = blur_mat(super_size, blur_sigma)
        dec_mat = dec_mat_func(super_size)
        #print('trans')
        #print(trans_mat)
        #print('---------\nblur')
        #print(blur_mat1)
        #print('--------\ndec')
        #print(dec_mat)
        #print(dec_mat)
        cur_lhs = dec_mat @ blur_mat1 @ trans_mat
        #print(cur_lhs)
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
    #print('-------\ntranspose mat')
    #print(transpose_mat)
    #print('--------\ntrans y 0')
    #print(trans_mat_y(super_size, offsets[0]))
    #print('--------\ntrans y 1')
    #print(trans_mat_y(super_size, offsets[1]))
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
    return coo_matrix((np.ones_like(input_pix_ind).flatten(), (output_pix_ind.flatten(), input_pix_ind.flatten())))


def blur_mat(super_size, blur_sigma):
    """
    Creates a blurring operator.
    """
    transpose_mat = transpose_mat_func(super_size)
    #print('transpose mat')
    #print(transpose_mat)
    #print('----------\nblur y')
    #print(blur_mat_y(super_size, blur_sigma))
    return (
            transpose_mat @ blur_mat_y(super_size, blur_sigma)
            @ transpose_mat @ blur_mat_y(super_size, blur_sigma)
    )


def blur_mat_y(super_size, blur_sigma):
    """
    Creates a blurring operator for the Y axis.
    """
    blur_kernel = GaussianKernel(np.arange(-1, 2), blur_sigma)
    #print('blur kernel')
    #print(blur_kernel)
    blur_kernel /= np.sum(blur_kernel)

    n = np.prod(super_size)
    row1 = np.zeros(n)
    row1[[-1, 0, 1]] = blur_kernel

    col1 = np.zeros(n)
    col1[0] = row1[0]
    col1[1] = row1[1]

    return sptoeplitz(col1, row1)


def dec_mat_func(super_size):
    """
    Creates a decimation operator.
    """
    sampled_size = (np.array(super_size) - 1) // 2
    #print(sampled_size)
    output_row, output_col = np.meshgrid(range(sampled_size[0]), range(sampled_size[1]))
    input_row = 2 * (output_row + 1)
    input_col = 2 * (output_col + 1)
    #print(input_row, output_row, input_col, output_col, sep='\n')
    input_ind = np.ravel_multi_index((input_col - 1, input_row - 1), super_size)
    output_ind = np.ravel_multi_index((output_col, output_row), sampled_size)
    #print(input_ind, output_ind, sep='\n')
    res = coo_matrix((np.ones_like(output_ind).flatten(), (output_ind.flatten(), input_ind.flatten())),
                      shape=(np.prod(sampled_size), np.prod(super_size)))
    #print(res)
    return res
