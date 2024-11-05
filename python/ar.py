import numpy as np
import cv2
from loadVid import loadVid
from planarH import compositeH, computeH_ransac
from matchPics import matchPics
from time import time

# HELPER FUNCTIONS
def crop_frame(frame: np.array, img: np.array, padding=45):
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

composite_frames = []

# open the log file

with open('ar_output.txt', 'w') as log_file:
    log_file.write(f'Starting Augmented Reality processing...\n\n')
    
    for i in range(book_frames_num):
        log_file.write(f'Starting frame-{i}\n')
        book_frame = book[i]
        frame = ar_source[i % ar_frames_num]
        resized_ar_frame = crop_frame(frame, cv_cover)
        
        log_file.write('Computing matches...\n')
        matches, locs1, locs2 = matchPics(cv_cover, book_frame)
        
        if matches.size == 0:
            log_file.write(f'No matches found for frame-{i}. Skipping...\n')
            continue
        
        locs1 = locs1[matches[:,0], 0:2]
        locs2 = locs2[matches[:,1], 0:2]
        
        log_file.write('Computing homography...\n')
        H2to1, inliers = computeH_ransac(locs1, locs2, iters=1000, thres=30)
        inliers_count = np.sum(inliers == 1)
        log_file.write(f'{matches.shape[0]} matches found, {inliers_count} inliers found for frame-{i}\n')
        
        warped_img = cv2.warpPerspective(resized_ar_frame, H2to1, dsize=(book_frame.shape[1], book_frame.shape[0]))
        composite_img = compositeH(H2to1, resized_ar_frame, book_frame)
        composite_frames.append(composite_img)
    
    writeVid(composite_frames, '../results/ar.avi', fps=30)

end_time = time()
duration = end_time - start_time
with open('ar_output.txt', 'a') as log_file:
    log_file.write(f'\nAugmented Reality processing finished in {duration:.3f} seconds.\n')
    
print(f'Augmented Reality code finished in {duration:.3f} seconds.')
