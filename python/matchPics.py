import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
import time

def matchPics(I1, I2, sigma=0.15, ratio=0.8):
    # adjust sigma and ratio
	# sigma = 0.15
	# ratio = 0.8

	# start timing
	start_time = time.time()
 
	# TODO: I1, I2 : Images to match

	# TODO: Convert Images to GrayScale
	I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    
	# TODO: Detect Features in Both Images
	locs1 = corner_detection(I1, sigma)
	locs2 = corner_detection(I2, sigma)
	
	# TODO: Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1, locs1)
	desc2, locs2 = computeBrief(I2, locs2)

	# TODO: Match features using the descriptors
	matches = briefMatch(desc1, desc2, ratio)
 
	end_time = time.time()
	duration = end_time - start_time
 
	print(f'matchPics: Finished computing {len(matches)} matches with sigma = {sigma} and ratio = {ratio} for {duration:.3f} seconds.')

	return matches, locs1, locs2
