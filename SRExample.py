import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse import diags
from scipy.ndimage import gaussian_filter
from skimage import io, color
from skimage.filters.rank import threshold
from skimage.transform import resize
import matplotlib.pyplot as plt
from SynthDataset import synth_dataset
from SREquations import sr_equations
from GradientDescent import GradientDescent
from ShowLRImages import show_lr_images

# Define your functions like SynthDataset, SREquations, GradientDescent, ShowLRImages before using them

# Prepare the reference image
im = io.imread('mira.jpeg')
if len(im.shape) == 3:  # If the image is RGB, convert to grayscale
    im = color.rgb2gray(im)
im = resize(im, (256, 256))  # Resize the image to 128x128
im = im.astype(np.float64)  # Convert to double precision
# Simulate the low-resolution images
num_images = 5
blur_sigma = 0.7
images, offsets, cropped_original = synth_dataset(im, num_images, blur_sigma)

# Compute the Super-Resolution image
lhs, rhs = sr_equations(images, offsets, blur_sigma)

#print(lhs, rhs)
#print(lhs)
col_sums = lhs.sum(axis=0)

K = diags(col_sums, [0], shape=(col_sums.shape[1], col_sums.shape[1]))
#print(K)
#print(lhs)
#print(K)
#B = scipy.sparse.csr_matrix(lhs.T @ rhs)
initial_guess = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(K), lhs.T @ rhs)  # This is an 'average' image produced from the LR images.

#print(K.shape, (lhs.T @ rhs).reshape(-1, 1).shape)
#initial_guess = np.linalg.solve(K.toarray(), (lhs.T @ rhs).reshape(-1, 1))

HR = GradientDescent(lhs, rhs, initial_guess, max_iter=185)
HR = HR.reshape(int(np.sqrt(HR.size)), -1)

# Visualize the results
show_lr_images(images)
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(HR, cmap='gray')
plt.title('Super-Resolution')

bicubic_interpolation = resize(resize(cropped_original, (cropped_original.shape[0] // 2, cropped_original.shape[1] // 2)), cropped_original.shape)
plt.subplot(1, 3, 2)
plt.imshow(bicubic_interpolation, cmap='gray')
plt.title('Bicubic Interpolation')

plt.subplot(1, 3, 3)
plt.imshow(cropped_original, cmap='gray')
plt.title('Reference')
plt.show()

# Compute the mean-square error of reconstruction
mse = np.mean((HR - cropped_original) ** 2)
print(f"Reconstruction Mean-square error: {mse}")