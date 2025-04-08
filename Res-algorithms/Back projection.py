import numpy as np
from scipy.interpolate import griddata
import cv2
import scipy
import math

def synth_dataset(im, sigma, num_images, scale=1):
    rows, cols = im.shape

    # Calculate the cropping indices
    working_row_sub = np.arange(0,  rows)
    working_col_sub = np.arange(0, cols)
    images = []
    offsets = np.zeros((num_images, 2))

    # Add the first image (no translation)
    offsets[0, :] = [0, 0]
    images.append(im)
    # Generate random translations for additional images
    for i in range(1, num_images):
        off = 2 * np.random.rand(1) - 1
        # off = [my_offsets[i]]
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
    kernel = cv2.getGaussianKernel(ksize=3, sigma=sigma)
    kernel = kernel @ kernel.T
    # Apply Gaussian blur to each image
    for i in range(num_images):
        images[i] = scipy.ndimage.convolve(images[i], kernel)

        # Downsample the image by a factor of 2
        cur_im = images[i]
        images[i] = cur_im[0::scale, 0::scale]

    return images, offsets

def simulate_lr(f_n, sigma, offsets, scale=1):
    lr_images = []
    rows, cols = f_n.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    kernel = cv2.getGaussianKernel(ksize=3, sigma=sigma)
    kernel = kernel @ kernel.T
    for i in range(len(offsets)):
        x2, y2 = np.meshgrid(np.arange(cols) - offsets[i][0], np.arange(rows) - offsets[i][1])
        translated_image = griddata(
            (x.ravel(), y.ravel()), f_n.ravel(), (x2, y2), method='linear', fill_value=0.5)
        blurred_image = scipy.ndimage.convolve(translated_image, kernel)
        blurred_image = blurred_image[0::scale, 0::scale]
        lr_images.append(blurred_image)
    return lr_images

def error_func(lr_real, lr_sim):
    lr_real = np.array(lr_real)
    lr_sim = np.array(lr_sim)
    return np.sqrt(np.sum((lr_real - lr_sim)**2))

def gaussian_psf(pix, sigma):
    return 1/(2*math.pi*sigma**2) * np.exp(-np.sum(pix**2) / (2*sigma**2))

def back_projection(lr_real, f_n, sigma, offsets, scale=1, normal = 2):
    lr_k = np.array(simulate_lr(f_n, sigma, offsets, scale))
    error_cur = error_func(lr_real, lr_k)
    error_prev = float('inf')
    lr_real = np.array(lr_real)
    iter = 0
    kernel = cv2.getGaussianKernel(ksize=3, sigma=sigma)
    kernel = kernel @ kernel.T
    denominator = normal * np.sum(np.sqrt(kernel))
    while abs(error_cur - error_prev) > 5 and iter < 20:
        error_prev = error_cur
        #for x, y in f_n:
            #f_n[x][y] +=
        f_n = f_n + scipy.ndimage.convolve(cv2.resize(np.sum(lr_real - lr_k, axis=0), f_n.shape), kernel)/ denominator
        lr_k = np.array(simulate_lr(f_n, sigma, offsets, scale))
        error_cur = error_func(lr_real, lr_k)
        iter += 1
        print("iter:", iter, "error:", error_cur, "delta:", error_prev - error_cur)
    return f_n

if __name__ == "__main__":
    image = cv2.imread('mira.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = 1
    scale = 2
    lrs, offsets = synth_dataset(image, sigma=sigma, num_images=4, scale=scale)
    init = cv2.resize(lrs[1], image.shape, interpolation=cv2.INTER_LINEAR)
    HR = back_projection(lrs, f_n = init, sigma=sigma, offsets=offsets, scale=scale)
    HR[HR < 0.01] = 0
    norm_lr = (lrs[1] * 255 / np.max(lrs[1])).astype(np.uint8)
    norm_HR = (HR * 255 / np.max(HR)).astype(np.uint8)
    cv2.imwrite('HR.png', norm_HR)
    cv2.imwrite('LR.png', norm_lr)
