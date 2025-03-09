import random

import numpy as np
import scipy.sparse.linalg
from scipy.sparse import diags
from skimage import io, color

from SREquations import sr_equations
from GradientDescent import GradientDescent

import cv2
from pathlib import Path
from random import randint


directory = Path("venus_shots/frames")
images = [cv2.imread(str(file)) for file in directory.glob("*.png")][:45]
for i in range(0, len(images)):
    images[i] = color.rgb2gray(images[i])
    images[i] = images[i].astype(np.float64)
    #images[i] -= np.mean(images[i])

offsets = [ [0, 0] for _ in range(len(images))]
# Compute the Super-Resolution image
sigma = 1/2
lhs, rhs = sr_equations(images, offsets, sigma)
col_sums = lhs.sum(axis=0)
K = diags(col_sums, [0], shape=(col_sums.shape[1], col_sums.shape[1]))
initial_guess = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(K), lhs.T @ rhs)
HR = GradientDescent(lhs, rhs, initial_guess, max_iter=95)
HR = HR.reshape(int(np.sqrt(HR.size)), -1)

norm_super = (np.abs(HR) * 255 / np.max(np.abs(HR))).astype(np.uint8)
cv2.imwrite(f'super_venus_{sigma}_45nz.jpg', norm_super)