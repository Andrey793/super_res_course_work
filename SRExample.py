import numpy as np
import scipy.sparse.linalg
from scipy.sparse import diags
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt
from SynthDataset import synth_dataset
from SREquations import sr_equations
from GradientDescent import GradientDescent
from ShowLRImages import show_lr_images
import cv2


# Prepare the reference image
im = io.imread('mira.jpeg')
if len(im.shape) == 3:  # If the image is RGB, convert to grayscale
    im = color.rgb2gray(im)
im = resize(im, (512, 512))  # Resize the image
im = im.astype(np.float64)  # Convert to double precision
# Simulate the low-resolution images
num_images = 5
blur_sigma = 0.6
images, offsets, cropped_original = synth_dataset(im, num_images, blur_sigma)

# Compute the Super-Resolution image
lhs, rhs = sr_equations(images, offsets, blur_sigma)

col_sums = lhs.sum(axis=0)

K = diags(col_sums, [0], shape=(col_sums.shape[1], col_sums.shape[1]))
initial_guess = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(K), lhs.T @ rhs)  # This is an 'average' image produced from the LR images.

HR = GradientDescent(lhs, rhs, initial_guess, max_iter=185)
HR = HR.reshape(int(np.sqrt(HR.size)), -1)

# Visualize the results
show_lr_images(images)
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(HR, cmap='gray')
HR[HR < 0.01] = 0
plt.title('Super-Resolution')
norm_super = (HR * 255 / np.max(HR)).astype(np.uint8)
cv2.imwrite('super.jpg', norm_super)

bicubic_interpolation = resize(resize(cropped_original, (cropped_original.shape[0] // 2, cropped_original.shape[1] // 2)), cropped_original.shape)
plt.subplot(1, 3, 2)
plt.imshow(bicubic_interpolation, cmap='gray')
plt.title('Bicubic Interpolation')

plt.subplot(1, 3, 3)
plt.imshow(cropped_original, cmap='gray')
plt.title('Reference')
plt.show()
norm = (cropped_original * 255 / np.max(cropped_original)).astype(np.uint8)
cv2.imwrite('reference.png', norm)

# Compute the mean-square error of reconstruction
mse = np.mean((HR - cropped_original) ** 2)
print(f"Reconstruction Mean-square error: {mse}")