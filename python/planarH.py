import numpy as np
import cv2

def computeH(x1, x2):
	# Q2.2.1
	# TODO: Compute the homography between two sets of points

	# x1, x2: N x 2 matrixes
	# x1 = H x2
	# when calculating H, we are actually solving A * h = 0, and A is the matrix from the coordinates of the two images

	N = len(x1)
	A_elements = np.zeros((N*2, 9))
 
	if N < 4:
		raise ValueError("At least 4 points are required to compute homography.")

	# we must now write the points as 2 x 9 matrices, so the A matrix will be (N x 2) x 9 matrices
	for i in range(N):
		# indices [i][0] account for the x-coordinate, while indices [i][1] account for the y-coordinate
		x1_x, x1_y = x1[i]	
		x2_x, x2_y = x2[i] 

		row_1 = [x2_x, x2_y, 1, 0, 0, 0, -x1_x * x2_x, -x1_x * x2_y, -x1_x]
		row_2 = [0, 0, 0, x2_x, x2_y, 1, -x1_y * x2_x, -x1_y * x2_y, -x1_y]
		A_elements[2 * i] = row_1
		A_elements[2 * i + 1] = row_2

	A = A_elements

	# the next step should be calculating eigenvalues, but we can't be sure to calculating eigenvalues when our matrix could be not square-shaped
	# thus, we will use the SVD approach, where matrix A = USV
	# U = left singular vectors, in which the columns are A(A^T)
	# S = diagonal matrix, in which each diagonal element is the singular value
	# V = right singular vectors, in which the columns are (A^T)A
	# but in python:
	# u = left singular vectors of (A^T)A
	# s = singular values of (A^T)A
	# v = right singular vectors of (A^T)A, returned as v^T
 
	_, _, v_T = np.linalg.svd(A.T @ A)

	# get the least-square solution of Ah = 0 which is in column 9 of matrix V
	# remember that previously we got v_T, so we have to transpose that again
	H2to1 = np.reshape(v_T.T[:, 8], (3, 3))
	print(f'computeH: finished with H2to1: {H2to1}')
	return H2to1


def computeH_norm(x1, x2):
	# Q2.2.2
	# TODO: Compute the centroid of the points
	x1_centroid = np.mean(x1, axis = 0)
	x2_centroid = np.mean(x2, axis = 0)

	# TODO: Shift the origin of the points to the centroid
	x1_moved = x1 - x1_centroid
	x2_moved = x2 - x2_centroid

	# TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	x1_scale = np.sqrt(2) / (np.max(np.linalg.norm(x1_moved,axis=1)))
	x2_scale = np.sqrt(2) / (np.max(np.linalg.norm(x2_moved,axis=1)))
	x1_norm = x1_scale * x1_moved
	x2_norm = x2_scale * x2_moved

	# TODO: Similarity transform 1: scale and shift (translate) the origin
	T1 = np.array([
		[x1_scale, 0, -x1_scale * x1_centroid[0]],
		[0, x1_scale, -x1_scale * x1_centroid[1]],
		[0, 0, 1]
	])

	# TODO: Similarity transform 2
	T2 = np.array([
		[x2_scale, 0, -x2_scale * x2_centroid[0]],
		[0, x2_scale, -x2_scale * x2_centroid[1]],
		[0, 0, 1]
	])

	# TODO: Compute homography
	H2to1_normalized = computeH(x1_norm, x2_norm)

	# TODO: Denormalization
	T1_inv = np.linalg.inv(T1)
	H2to1 = T1_inv @ H2to1_normalized @ T2
	print(f'computeH_norm: finished with H2to1: {H2to1}')
	return H2to1


def computeH_ransac(locs1, locs2):
	# Q2.2.3
	# TODO: Compute the best fitting homography given a list of matching points
 
	N = len(locs1)
	inliers = np.zeros((N, 1))
	bestH2to1 = np.zeros((3,3))
 
	# variables
	iters = 1000 # number of RANSAC iterations
	thres = 2.0 # threshold to determine inliers
 
	for i in range(iters):
		# pick 4 random points
		idx = np.random.randint(0, N, size = 4)
		x1 = locs1[idx]
		x2 = locs2[idx]
  
		# compute homography
		H2to1 = computeH_norm(x1, x2)
		temp_inliers = np.zeros((N, 1))
  
		# transform the points so we can see where points in locs2 would appear in locs1
		for j in range(N):
			# convert locs1 point to homogeneous coordinates
			x1_homogenous = np.hstack((locs1[j] , 1))

			# apply homography transformation
			x2_calc = np.dot(H2to1, x1_homogenous)
			x2_calc = x2_calc / x2_calc[2] # normalize to make homogenous
			x2_calc = x2_calc[:2]
   
			# calculate euclidean distance between projection and original point in locs2
			l2_dist = np.linalg.norm(locs2[j] - x2_calc)

			if l2_dist < thres:
				temp_inliers[j] = 1
    
		if np.sum(temp_inliers) > np.sum(inliers):
			inliers = temp_inliers
			bestH2to1 = H2to1
	print(f'computeH_ransac: finished with bestH2to1: {bestH2to1} and inliers: {inliers}\n and iters: {iters} and threshold: {thres}')
	return bestH2to1, inliers


def compositeH(H2to1, template, img):
    # create a mask of the same size as the template
    mask = np.ones(template.shape[:2], dtype=np.uint8) * 255

    # warp the mask with the inverse homography
    warp_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))

    # warp the template image with the homography
    warp_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

    # Combine the images using the mask
    img_background = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(warp_mask))
    img_foreground = cv2.bitwise_and(warp_template, warp_template, mask=warp_mask)

    # Combine the foreground and background images
    composite_img = cv2.add(img_background, img_foreground)

    return composite_img

