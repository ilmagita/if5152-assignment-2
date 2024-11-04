import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

# matches, locs1, locs2 = matchPics(cv_cover, cv_desk, sigma=0.2, ratio=0.9)

for i in np.arange(0.1, 2, 0.1):
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, sigma=0.15, ratio=i)

# display matched features
# plotMatches(cv_cover, cv_desk, matches, locs1, locs2)
