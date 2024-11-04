import numpy as np
import cv2

def computeH(x1, x2):
    # Q2.2.1
    # TODO: Compute the homography between two sets of points
    
    # INPUTS
    # x1 : N x 2 matrix; coordinates of point pairs between two images in image 1
    # x2 : N x 2 matrix; coordinates of point pairs between two images in image 2
    
    # OUTPUTS
    # H2to1 : 3 x 3 matrix; best homography from image 2 to 1 in the least-square sense
    
    x1 = x1.T
    x2 = x2.T
    
    N = len(x1)
    
    if N < 4:
        raise ValueError('At least 4 points required to compute homography.')
    
    A = np.empty([2 * N, 9])
    for i in range(N):
        x1_xi = x1[i, 0] # x coordinate in x1
        x1_yi = x1[i, 1] # y coordinate in x1
        x2_xi = x2[i, 0] # x coordinate in x2
        x2_yi = x2[i, 1] # y coordinate in x2
        A[2*i] = [x2_xi, x2_yi, 1, 0, 0, 0, -x1_xi * x2_xi, -x1_xi * x2_yi, -x1_xi]
        A[2*i + 1] = [0, 0, 0, x2_xi, x2_yi, 1, -x1_yi * x2_xi, -x1_yi * x2_yi, -x1_yi]
    
    _, _, V_t = np.linalg.svd(A.T @ A)
    smallest_eigenvector = V_t[-1, :]
    H2to1 = smallest_eigenvector.reshape(3,3)
    return H2to1

def computeH_norm(x1, x2):
	# Q2.2.2
	# TODO: Compute the centroid of the points
 
	# INPUTS
    # x1 : N x 2 matrix; coordinates of point pairs between two images in image 1
    # x2 : N x 2 matrix; coordinates of point pairs between two images in image 2
	
	# OUTPUTS
	# H2to1 : 3 x 3 matrix; best homography from image 2 to 1 in the least-square sense
 
	# TODO: Shift so centroid is at the origin (NOTE: Done in similarity transform!)
	# TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
 
	# print('Starting computeH_norm')
	# print(f'computeH_norm >> x1:\n{x1}')
	# print(f'computeH_norm >> x2:\n{x2}')
    
	mean_x1_x = np.mean(x1[:,0])
	mean_x1_y = np.mean(x1[:,1])
	mean_x2_x = np.mean(x2[:,0])
	mean_x2_y = np.mean(x2[:,1])
 
	N1 = x1.shape[0]
	N2 = x2.shape[0]
	s_x1 = np.empty((N1))
	s_x2 = np.empty((N2))

	# Get the scaling factor for both x1 and x2
	for i in range(N1):
		s_x1[i] = np.sqrt((x1[i, 0] - mean_x1_x) ** 2 + (x1[i, 1] - mean_x1_y) ** 2)
	x1_scale = np.sqrt(2)/np.amax(s_x1)
 
	for i in range(N2):
		s_x2[i] = np.sqrt((x2[i, 0] - mean_x2_x) ** 2 + (x1[i, 1] - mean_x1_y) ** 2)
	x2_scale = np.sqrt(2)/np.amax(s_x2)

	# TODO: Similarity transform 1: scale and shift (translate) the origin
	# T1 = scaling matrix (dot) translation matrix
	T1 = np.array([
		[x1_scale, 0, -x1_scale * mean_x1_x],
		[0, x1_scale, -x1_scale * mean_x1_y],
		[0, 0, 1]
	])
 
	# x1_hom = np.hstack((x1, np.ones((len(x1), 1))))
	# print(f'computeH_norm >> x1_hom:\n{x1_hom}')
	# x1_hom = T1 @ x1_hom.T
	# print(f'computeH_norm >> x1_hom after transformed:\n{x1_hom}')
	x1_normalized = np.hstack((x1, np.ones((N1, 1))))
	x1_hom = T1 @ x1_normalized.T

	# TODO: Similarity transform 2
	T2 = np.array([
		[x2_scale, 0, -x2_scale * mean_x2_x],
		[0, x2_scale, -x2_scale * mean_x2_y],
		[0, 0, 1]
	])
	# x2_hom = np.hstack((x2, np.ones((len(x1), 1))))
	# print(f'computeH_norm >> x2_hom: {x2_hom}')
	# x2_hom = T2 @ x2_hom.T
	# print(f'computeH_norm >> x2_hom after transformed:\n{x2_hom}')
	x2_normalized = np.hstack((x2, np.ones((N2, 1))))
	x2_hom = T2 @ x2_normalized.T
 
	# TODO: Compute homography
	# H2to1_normalized = computeH(x1_hom, x2_hom)
	H2to1_normalized = computeH(x1_hom, x2_hom)
	# print(f'computeH_norm >> H2to1_normalized:\n{H2to1_normalized}')

	# TODO: Denormalization
	H2to1 = np.linalg.inv(T2) @ H2to1_normalized @ T1
	# print(f'computeH_norm >> finished with H2to1: {H2to1}')
	return H2to1

def computeH_ransac(locs1, locs2, iters=700, thres=2.0):
	# Q2.2.3
	# TODO: Compute the best fitting homography given a list of matching points
 
 	# INPUTS
	# locs1 : N x 2 matrix, each row has (x, y) of a feature point
	# locs2 : N x 2 matrix, each row has (x, y) of a feature point
	# iters : max number of iterations
	# thres : threshold to determine inliers
 
	# OUTPUTS
	# bestH2to1 : best homography designated by ransac algorithm
	# inliers : vector of length N with a 1 at matches
 
	N = len(locs1)
	inliers = np.zeros((1, N))
	bestH2to1 = np.zeros((3,3))
 
	for i in range(iters):
		# pick 4 random points
		idx = np.random.choice(N, size=4, replace=False)
		x1 = locs1[idx]
		x2 = locs2[idx]
		# print(f'computeH_ransac >> x1 chosen: {x1}')
		# print(f'computeH_ransac >> x2 chosen: {x2}')
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
			inliers = temp_inliers.T
			bestH2to1 = H2to1
   
	# print(f'computeH_ransac: finished with bestH2to1: {bestH2to1} and inliers: {inliers}\n and iters: {iters} and threshold: {thres}\n and inliers_count:{np.sum(inliers == 1)}')
	return bestH2to1, inliers

def compositeH(H2to1, template, img):
    # Composites two images
    
    # INPUTS
    # H2to1 : 3 x 3 matrix containing homography
    # template : template image that will be projected
    # img : image that will have projected image on top
    
    # OUTPUTS
    # composite_img : template on top of original image
    
    # create a mask of the same size as the template
    mask = np.ones(template.shape[:2], dtype=np.uint8) * 255

    # warp the mask with the inverse homography
    warp_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))

    # warp the template image with the homography
    warp_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

    # combine the images using the mask
    img_background = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(warp_mask))
    img_foreground = cv2.bitwise_and(warp_template, warp_template, mask=warp_mask)

    # combine the foreground and background images
    composite_img = cv2.add(img_background, img_foreground)

    return composite_img

