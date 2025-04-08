import cv2
import numpy as np
import glob
import os


# Load all images
image_folder = "/home/andrey/HSE/Course work/Moon_dataset/moons_centerd"
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.tif")))

# Read all images
images = [cv2.imread(img) for img in image_paths]

# Convert images to grayscale
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

sift = cv2.SIFT_create()

keypoints_descriptors = [sift.detectAndCompute(img, None) for img in gray_images]

# Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=150)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# reference
base_img = images[0]
base_kp, base_desc = keypoints_descriptors[0]

# Aligning
aligned_images = [base_img]

for i in range(1, len(images)):
    kp, desc = keypoints_descriptors[i]

    matches = flann.knnMatch(base_desc, desc, k=2)

    #ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 130:
        src_pts = np.float32([base_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        height, width, _ = base_img.shape
        aligned = cv2.warpPerspective(images[i], H, (width, height))
        aligned_images.append(aligned)


output_folder = "aligned_images"
os.makedirs(output_folder, exist_ok=True)

for i, img in enumerate(aligned_images):
    cv2.imwrite(os.path.join(output_folder, f"aligned_{i}.jpg"), img)
