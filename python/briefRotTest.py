import numpy as np
import cv2
from matchPics import matchPics

import skimage.color
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def plotMatches(im1,im2,matches,locs1,locs2, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.axis('off')
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r',only_matches=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)

#Q2.1.5
#Read the image and convert to grayscale, if necessary
I = cv2.imread('../data/cv_cover.jpg')

matches_count = []
rotations = []

for i in range(36):
	#Rotate Image
	rotations.append(i * 10)
	I_rotated = rotate(I, i*10, reshape=False)
 
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(I, I_rotated)
 
	if (i * 10) in [0, 10, 350]:
		output_path = (f'../results/plotMatches_rotation_{i * 10}.jpg')
		plotMatches(I, I_rotated, matches, locs1, locs2, output_path)

	#Update histogram
	matches_count.append(len(matches))
	print(f'Matching at rotation {i*10} degrees done.')

#Display histogram
plt.bar(rotations, matches_count, color='blue', alpha=0.6)
plt.xlabel("Degree of Rotations")
plt.ylabel("Matches Count")
plt.title("Rotations vs. Number of Matches")
plt.show()
