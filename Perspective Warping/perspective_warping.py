# ======= imports
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt

# ======= constants
figsize = (10, 10)
feature_extractor = cv2.SIFT_create()
bf = cv2.BFMatcher()

def warpTwoImages(img1, img2, H):
   
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(img2, Ht@H, (xmax - xmin, ymax - ymin))
    
    # result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    for i in range(h1):
        for j in range(w1):
            if result[i][j].max() == 0:
                result[i][j] = img1[i][j]   
    # cv2.imshow('frame', result)
    #plt.title('result')
    # plt.show()
    return result


# === template image keypoint and descriptors
img_template = cv2.imread('template.png')
img_template_rgb = cv2.cvtColor(img_template, cv2.COLOR_BGR2RGB)
img_template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

# ===== video input, output and metadata
cap = cv2.VideoCapture('newinput.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

img_input = cv2.imread('img_input.jpg')
img_input_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
img_input = cv2.resize(img_input_rgb, (img_template.shape[1], img_template.shape[0]))

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# ========== run on all frames
while True:
    cv2.resizeWindow('frame', 960, 540)
    ret, frame = cap.read()
 
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    farame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_farme, desc_frame = feature_extractor.detectAndCompute(farame_gray, None)
    kp_img_template, desc_img_template = feature_extractor.detectAndCompute(img_template_gray, None)

    matches = bf.knnMatch(desc_frame, desc_img_template, k=2)
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance/m[1].distance < 0.5:
         good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]

    # ======== find homography
    # also in SIFT notebook
    good_kp_template = np.array([kp_img_template[m.trainIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([kp_farme[m.queryIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)

    # ++++++++ do warping of another image on template image
    # we saw this in SIFT notebook
    result = warpTwoImages(frame_rgb, img_input, H)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # img_warped = cv2.warpPerspective(img_input, H, (frame_rgb.shape[1], frame_rgb.shape[0]))
    # mask = np.where(img_warped == 0)
    # img_warped[mask] = frame_rgb[mask]
    # result = img_warped

    # =========== plot and save frame
    out.write(result)
    cv2.imshow('frame', result)

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

# ======== end all
cap.release()
out.release()
cv2.destroyAllWindows()
