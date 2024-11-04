import numpy as np
import cv2
from matchPics import matchPics
import matplotlib.pyplot as plt
import skimage.feature
# from helper import plotMatches

def plotMatches(im1,im2,matches,locs1,locs2, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.axis('off')
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r',only_matches=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

for ratio in [0.7]:
    for i in [0.05]:
        try:
            matches, locs1, locs2 = matchPics(cv_cover, cv_desk, sigma=i, ratio=ratio)
            if len(matches) > 0:
                output_path = (f'../results/plotMatches_ratio_{ratio}_sigma_{i:.2f}.jpg')
                plotMatches(cv_cover, cv_desk, matches, locs1, locs2, output_path)
        except IndexError as e:
            print(f'IndexError occurred for ratio={ratio} and sigma={i}: {e}')
            continue