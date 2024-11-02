import numpy as np
import cv2
from matchPics import matchPics

import skimage.color
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

#Q2.1.5
#Read the image and convert to grayscale, if necessary
I = cv2.imread('data/cv_cover.jpg')
if len(I.shape) == 3:
	I = skimage.color.rgb2gray(I)

matches_count = []
rotations = []

for i in range(36):
	#Rotate Image
	rotations.append(i * 10)
	I_rotated = rotate(I, i*10, reshape=False)
 
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(I, I_rotated)
 
	#Update histogram
	matches_count.append(len(matches))
	print(f'Matching at rotation {i*10} degrees done.')

#Display histogram
plt.bar(rotations, matches_count, density=True, color='blue', alpha=0.6)
plt.xlabel("Count of Matches")
plt.ylabel("Rotation in Degrees")
plt.title("Rotations vs. Number of Matches")
plt.show()
