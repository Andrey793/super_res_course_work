import numpy as np
from scipy.sparse import diags
from skimage import color

from SREquations import sr_equations
import cv2
from pathlib import Path
import GradientDescent
import scipy

directory = Path("/home/andrey/HSE/Course work/Moon_dataset/aligned_images4/")
images = [cv2.imread(str(file)) for file in directory.glob("*.jpg")][:9]
for i in range(0, len(images)):
    images[i] = color.rgb2gray(images[i])
    images[i] = images[i].astype(np.float64)

offsets = [ [0, 0] for _ in range(len(images))]
# Compute the Super-Resolution image
for sigma in np.arange(0.2, 0.3, 0.15):
    lhs, rhs = sr_equations(images, offsets, sigma)
    initial_guess = cv2.resize(images[0], (int(np.sqrt(lhs.shape[1])), int(np.sqrt(lhs.shape[1]))))
    norm_init = (np.abs(initial_guess) * 255 / np.max(np.abs(initial_guess))).astype(np.uint8)
    initial_guess = initial_guess.flatten()
    res = scipy.sparse.linalg.lsqr(lhs, rhs, iter_lim=3000, x0=initial_guess)

    HR = res[0]
    istop = res[1]
    itn = res[2]
    r1norm = res[3]
    x_k_norm = res[8]
    print("sigma: ", sigma, "-" * 55)
    print("istop", istop)
    print("r1norm", r1norm)
    print("itn", itn)

    HR = HR.reshape(int(np.sqrt(HR.size)), -1)

    norm_super = (np.abs(HR) * 255 / np.max(np.abs(HR))).astype(np.uint8)
    cv2.imwrite(f'moon_gradient/super_moonX2_8_3_{sigma}.png', norm_super)