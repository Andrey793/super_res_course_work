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
    #images[i] -= np.mean(images[i])

offsets = [ [0, 0] for _ in range(len(images))]
# Compute the Super-Resolution image
for sigma in np.arange(0.3, 1.2, 1.22):
    lhs, rhs = sr_equations(images, offsets, sigma)
    col_sums = lhs.sum(axis=0)
    K = diags(col_sums, [0], shape=(col_sums.shape[1], col_sums.shape[1]))
    #initial_guess = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(K), lhs.T @ rhs)
    print(np.sqrt(lhs.shape[1]))
    initial_guess = cv2.resize(images[0], (int(np.sqrt(lhs.shape[1])), int(np.sqrt(lhs.shape[1]))))
    norm_init = (np.abs(initial_guess) * 255 / np.max(np.abs(initial_guess))).astype(np.uint8)
    #cv2.imwrite(f'kacmarz/initial_venus.jpg', norm_init)
    initial_guess = initial_guess.flatten()
    #HR = GradientDescent(lhs, rhs, initial_guess, max_iter=95)
    #HR = conjugate_gradient(lhs, rhs, initial_guess, max_iter=100)
    #HR = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    res = scipy.sparse.linalg.lsqr(lhs, rhs, iter_lim=3000, show=True, x0=initial_guess)
    HR = res[0]
    istop = res[1]
    itn = res[2]
    r1norm = res[3]
    x_k_norm = res[8]
    print("istop", istop)
    print("r1norm", r1norm)
    print("itn", itn)
    #print(images[0].flatten().shape)
    #print(lhs.shape)
    #A_TA = csc_matrix(lhs.T @ lhs)  # Square matrix
    #A_Tb = lhs.T @ rhs
    #res, info = scipy.sparse.linalg.bicgstab(A_TA, A_Tb, rtol=1e-19, callback=save_xk, x0=initial_guess, maxiter=1000)
    #print(np.linalg.norm(A_Tb))
    #print("info", info)
    #HR = res
    #print(np.log(log_diff))
    #print(np.log(log_x_ks))
    #print(np.log(np.linalg.norm(res)))
    #print(np.linalg.norm(A_TA@res - A_Tb))

    #plt.plot(np.log(log_diff), np.log(log_x_ks), '-bo')
    #for i in range(len(log_x_ks)):
    #    plt.text(np.log(log_diff[i]), np.log(log_x_ks[i]), str(i), fontsize=12, ha='right', va='bottom', color='red')
    #plt.xlim(-30, 0)
    #plt.show()
    #HR = kaczmarz.Cyclic.solve(lhs, rhs, initial_guess)
    '''
    rows_i, cols_i = lhs.nonzero()
    n = 0
    for i, j in zip(rows_i, cols_i):
        if i == j:
            n += 1
    '''
    #print("Non zero diag elements: ", n)
    #HR = GradientDescent.kaczmarz(lhs, rhs, tol=1e-07, x0=initial_guess, max_iter=12000)
    #print("sigma:", sigma, "info:", info)
    #print("sigma:", sigma, 'error:', res[3], "iterations:", res[2], "istop:", res[1])

    HR = HR.reshape(int(np.sqrt(HR.size)), -1)

    norm_super = (np.abs(HR) * 255 / np.max(np.abs(HR))).astype(np.uint8)
    cv2.imwrite(f'lsqr/super_venus_{sigma}_150_08_v2.jpg', norm_super)