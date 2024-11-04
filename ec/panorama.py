import numpy as np
import cv2
import skimage.color
from planarH import compositeH
from planarH import computeH_ransac
from matchPics import matchPics
from time import time

# MAIN PROGRAM
print('Starting panorama.py')
start_time = time()

# Load Images
pano_left = cv2.imread('../data/kos_left.jpg')
pano_right = cv2.imread('../data/kos_right.jpg')

im1_H, im1_W, _ = pano_left.shape
im2_H, im2_W, _ = pano_right.shape
width = round(max(im2_W, im1_W) * 1.2)

adjusted_right = cv2.copyMakeBorder(pano_right, 0, im2_H - im1_H, width - im2_W, 0, cv2.BORDER_CONSTANT, 0)

# Select Matching Points
matches, locs1, locs2 = matchPics(pano_left, adjusted_right, sigma=3.0, ratio=0.8)
# plotMatches(pano_left, adjusted_right, matches, locs1, locs2)

locs1 = locs1[matches[:, 0], 0:2]
locs2 = locs2[matches[:, 1], 0:2]

# Stitch Images Together
bestH2to1, _ = computeH_ransac(locs1, locs2, iters=1000, thres=4.0)
panorama_stitched = compositeH(bestH2to1, pano_left, adjusted_right)
panorama = np.maximum(adjusted_right, panorama_stitched)

cv2.imwrite('results/panorama_stitched_own.jpg', panorama)

end_time = time()
print(f'and {len(matches)} matches.')
print(f'Finished panorama.py in {end_time - start_time} seconds.')