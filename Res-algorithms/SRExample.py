import numpy as np
import scipy.sparse.linalg
from scipy.sparse import diags
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
from SynthDataset import synth_dataset
from SREquations import sr_equations
from GradientDescent import GradientDescent
import cv2


# Prepare the reference image
im = io.imread('mira.jpg')
if len(im.shape) == 3:
    im = color.rgb2gray(im)
im = resize(im, (512, 512))
im = im.astype(np.float64)
# Simulate the low-resolution images
num_images = 5
blur_sigma = 1
images, offsets, cropped_original = synth_dataset(im, num_images, blur_sigma)

# Compute the Super-Resolution image
lhs, rhs = sr_equations(images, offsets, blur_sigma)
col_sums = lhs.sum(axis=0)

K = diags(col_sums, [0], shape=(col_sums.shape[1], col_sums.shape[1]))
initial_guess = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(K), lhs.T @ rhs)


HR = GradientDescent(lhs, rhs, initial_guess, max_iter=185)

HR = HR.reshape(int(np.sqrt(HR.size)), -1)

# Save the results
HR[HR < 0.01] = 0
norm_super = (HR * 255 / np.max(HR)).astype(np.uint8)
cv2.imwrite(f'super{num_images}_ML.jpg', norm_super)