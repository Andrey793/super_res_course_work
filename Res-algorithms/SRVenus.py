import numpy as np
import scipy.sparse.linalg
from scipy.sparse import diags, csc_matrix
from skimage import io, color

from SREquations import sr_equations
from GradientDescent import GradientDescent
from GradientDescent import conjugate_gradient
import cv2
from pathlib import Path
#import kaczmarz
import GradientDescent
import matplotlib.pyplot as plt
import Ninox_algo

log_x_ks = []
log_diff = []
def save_xk(xk):
    log_x_ks.append(np.linalg.norm(xk))
    log_diff.append(np.linalg.norm(A_TA @ xk - A_Tb))

directory = Path("venus_shots2/")
images = [cv2.imread(str(file)) for file in directory.glob("*.png")][:300]
for i in range(0, len(images)):
    images[i] = color.rgb2gray(images[i])
    images[i] = images[i].astype(np.float64)

offsets = [ [0, 0] for _ in range(len(images))]
# Compute the Super-Resolution image
for sigma in np.arange(0.3, 1.2, 1.22):
    lhs, rhs = sr_equations(images, offsets, sigma)

    initial_guess = cv2.resize(images[0], (int(np.sqrt(lhs.shape[1])), int(np.sqrt(lhs.shape[1]))))
    norm_init = (np.abs(initial_guess) * 255 / np.max(np.abs(initial_guess))).astype(np.uint8)
    initial_guess = initial_guess.flatten()
    res = scipy.sparse.linalg.lsqr(lhs, rhs, iter_lim=3000, show=True, x0=initial_guess)
    HR = res[0]
    istop = res[1]
    itn = res[2]
    r1norm = res[3]
    x_k_norm = res[8]
    print("istop", istop)
    print("r1norm", r1norm)
    print("itn", itn)

    HR = HR.reshape(int(np.sqrt(HR.size)), -1)

    norm_super = (np.abs(HR) * 255 / np.max(np.abs(HR))).astype(np.uint8)
    cv2.imwrite(f'lsqr/super_venus_{sigma}_150_08_v2.jpg', norm_super)