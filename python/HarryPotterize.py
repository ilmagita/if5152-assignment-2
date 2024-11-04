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
    print(f'Computing matches with ratio={ratio} and sigma={sigma}')
    # match it accordingly
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, sigma=sigma, ratio=ratio)
    print(f'Done doing matchPics with cv_desk and cv_cover with sigma={sigma} and ratio={ratio}')

    locs1 = locs1[matches[:,0], 0:2]
    locs2 = locs2[matches[:,1], 0:2]
    
    matches_count = len(matches)
    
    return ratio, sigma, locs1, locs2, matches_count

def HarryPotterize(ratio, sigma, locs1, locs2, iters, matches_count, thres):
    print(f'Starting HarryPotterize for Ratio: {ratio}, Sigma: {sigma}, Iters: {iters}, Inlier tolerance: {thres}')

    H2to1, inliers = computeH_ransac(locs1, locs2, iters=iters, thres=thres)

    # warps hp_cover.jpg to the dimension of cv_desk.png using skimage function skimage.transform.warp
    warped_img = cv2.warpPerspective(hp_cover_resized, H2to1, dsize=(cv_desk.shape[1], cv_desk.shape[0]))
    cv2.imwrite(f'../results/HarryPotter/projected_r{ratio}_s{sigma}_i{iters}_t{thres}.jpg', warped_img)

    composite_img = compositeH(H2to1, hp_cover_resized, cv_desk)
    cv2.imwrite(f'../results/HarryPotter/desk_r{ratio}_s{sigma}_i{iters}_t{thres}.jpg', composite_img)
    print(f'HarryPotterize done for ratio={ratio}, sigma={sigma}, matches={matches_count}, iters={iters}, thres={thres}.')
    
## VARIABLES
ratio_arr = [0.5, 0.5]
sigma_arr = [0.06, 0.07]

iters_arr = [2000, 2000, 2000, 2000]
thres_arr = [50, 30, 15, 10]

for i, rat in enumerate(ratio_arr):
    ratio, sigma, locs1, locs2, matches_count = getMatches(rat, sigma_arr[i])
    for j, it in enumerate(iters_arr):
        HarryPotterize(ratio, sigma, locs1, locs2, it, matches_count, thres_arr[j])
    
print('Ending HarryPotterize.py')
end_time = time()
print(f'Finished in {end_time - start_time} seconds')