import numpy as np
import cv2

# TODO: Import necessary functions
from loadVid import loadVid
from planarH import compositeH, computeH_ransac
from matchPics import matchPics
from time import time

# TODO: Write script for Q3.1

# HELPER FUNCTIONS
def crop_frame(frame: np.array, img: np.array, padding=45):
    # INPUTS
    # frame : array with 2-3 columns
    # img : array with 2-3 columns
    # padding : amount of padding to be done
    
    # OUTPUTS
    # resized_frame : frame that has been resized to match the aspect ratio of img
    
    img_H, img_W = img.shape[0], img.shape[1]
    frame_H, frame_W = frame.shape[0], frame.shape[1]
    
    new_width = int(img_W * (frame_H / img_H))
    start_x = (frame_W - new_width) // 2
    cropped_frame = frame[padding: -padding, start_x:start_x + new_width]
    
    resized_frame = cv2.resize(cropped_frame, (img_W, img_H))
    
    return resized_frame

def writeVid(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

# MAIN PROGRAM
start_time = time()

ar_source = loadVid('../data/ar_source.mov')
book = loadVid('../data/book.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')

ar_frames_num = ar_source.shape[0]
ar_H = ar_source.shape[1]
ar_W = ar_source.shape[2]

book_frames_num = book.shape[0]
book_H = book.shape[1]
book_W = book.shape[2]

print(f'Number of frames in panda video: {ar_frames_num}') # 511
print(f'AR height: {ar_H}') # 360
print(f'AR width: {ar_W}') # 640
print(f'Type of AR video: {type(ar_source)}') # <class 'numpy.ndarray'>

print(f'Number of frames in book video: {book_frames_num}') # 641
print(f'Book height: {book_H}') # 480
print(f'Book width: {book_W}') # 640

# Track CV text book in each frame of book.mov,
# and overlay each frame of ar_source.mov onto the book in book.mov
# Don't forget to crop each frame to fit onto the book cover, so only its central region is used.

composite_frames = []

for i in range(10):
    print(f'Starting frame-{i}')
    book_frame = book[i]
    frame = ar_source[i % ar_frames_num]
    resized_ar_frame = crop_frame(frame, cv_cover)
    
    matches, locs1, locs2 = matchPics(cv_cover, book_frame)
    locs1 = locs1[matches[:,0], 0:2]
    locs2 = locs2[matches[:,1], 0:2]
    
    H2to1, inliers = computeH_ransac(locs1, locs2)
    
    warped_img = cv2.warpPerspective(resized_ar_frame, H2to1, dsize=(book_frame.shape[1], book_frame.shape[0]))
    composite_img = compositeH(H2to1, resized_ar_frame, book_frame)
    composite_frames.append(composite_img)
    
writeVid(composite_frames, '../results/ar.avi', fps=30)

end_time = time()
print(f'Augmented Reality code finished in {end_time - start_time:.3f} seconds.')