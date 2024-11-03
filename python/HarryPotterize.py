import numpy as np
import cv2
import skimage.io 
import skimage.color

# TODO: Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, computeH, computeH_norm
from planarH import compositeH

# TODO: Write script for Q2.2.4

# Reads cv_cover.jpg, cv_desk.png, hp_cover.jpg
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# Computes a homography automatically using MatchPics and computeH_ransac
# resize hp_cover
hp_cover_resized = cv2.resize(hp_cover, dsize=(cv_cover.shape[1], cv_cover.shape[0]))

# match it accordingly
matches, locs1, locs2 = matchPics(cv_desk, cv_cover)

locs1 = locs1[matches[:,0], 0:2]
locs2 = locs2[matches[:,1], 0:2]

H2to1, inliers = computeH_ransac(locs1, locs2)

# Warps hp_cover.jpg to the dimension of cv_desk.png using skimage function skimage.transform.warp
warped_img = cv2.warpPerspective(hp_cover_resized, H2to1, dsize=(cv_desk.shape[1], cv_desk.shape[0]))
cv2.imwrite('../results/HarryPotter_made.jpg', warped_img)
cv2.imshow('HarryPotter_made', warped_img)

composite_img = compositeH(H2to1, hp_cover_resized, cv_desk)
cv2.imwrite('../results/composite_harryPotter_desk.jpg', composite_img) 

