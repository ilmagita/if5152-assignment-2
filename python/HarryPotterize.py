import numpy as np
import cv2
import skimage.io 
import skimage.color

# TODO: Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from time import time

# TODO: Write script for Q2.2.4
print('Starting HarryPotterize.py')
start_time = time()

# Reads cv_cover.jpg, cv_desk.png, hp_cover.jpg
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# Computes a homography automatically using MatchPics and computeH_ransac
# resize hp_cover
hp_cover_resized = cv2.resize(hp_cover, dsize=(cv_cover.shape[1], cv_cover.shape[0]))

def getMatches(ratio, sigma):
    matches_start_time = time()
    print(f'Computing matches with ratio={ratio} and sigma={sigma}')
    # match it accordingly
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, sigma=sigma, ratio=ratio)
    matches_count = len(matches)
    print(f'Done computing {matches_count} with sigma={sigma} and ratio={ratio}')

    locs1 = locs1[matches[:,0], 0:2]
    locs2 = locs2[matches[:,1], 0:2]
    
    matches_end_time = time()
    
    print(f'Done computing matches in {matches_end_time - matches_start_time:.3f} seconds.')
    
    return ratio, sigma, locs1, locs2, matches_count

def HarryPotterize(ratio, sigma, locs1, locs2, iters, matches_count, thres):
    hp_start_time = time()
    print(f'Starting HarryPotterize for Ratio: {ratio}, Sigma: {sigma}, Iters: {iters}, Inlier tolerance: {thres}')

    H2to1, inliers = computeH_ransac(locs1, locs2, iters=iters, thres=thres)

    # warps hp_cover.jpg to the dimension of cv_desk.png using skimage function skimage.transform.warp
    # warped_img = cv2.warpPerspective(hp_cover_resized, H2to1, dsize=(cv_desk.shape[1], cv_desk.shape[0]))
    # cv2.imwrite(f'../results/HarryPotter/projected/projected_r{ratio}_s{sigma}_i{iters}_t{thres}.jpg', warped_img)
    print(f'Number of inliers: {np.sum(inliers == 1)}')
    composite_img = compositeH(H2to1, hp_cover_resized, cv_desk)
    cv2.imwrite(f'../results/desk_r{ratio}_s{sigma}_i{iters}_t{thres}.jpg', composite_img)
    hp_end_time = time()
    print(f'HarryPotterize done in {hp_end_time - hp_start_time:.3f} seconds')
    
## VARIABLES
ratio_arr = [0.7]
sigma_arr = [0.15] # must be the same length as sigma_arr

iters_arr = [1000, 1000, 500, 500]
thres_arr = [50, 30, 50, 30]   # must be the same length as iters_arr

for i, rat in enumerate(ratio_arr):
    ratio, sigma, locs1, locs2, matches_count = getMatches(rat, sigma_arr[i])
    for j, it in enumerate(iters_arr):
        HarryPotterize(ratio, sigma, locs1, locs2, it, matches_count, thres_arr[j])
    
print('Ending HarryPotterize.py')
end_time = time()
print(f'Finished in {end_time - start_time} seconds')