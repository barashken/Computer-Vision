# ======= imports

import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import mesh_renderer

square_size =2.4

objectPoints = (
    3
    * square_size
    * np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
)


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img

# ======= constants
figsize = (10, 10)
feature_extractor = cv2.SIFT_create()
bf = cv2.BFMatcher()

# === template image keypoint and descriptors
img_template = cv2.imread('template1.png')
img_template_rgb = cv2.cvtColor(img_template, cv2.COLOR_BGR2RGB)
img_template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

# ===== video input, output and metadata
cap = cv2.VideoCapture('input.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('ARoutput.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

cali_video = cv2.VideoCapture("cali_video.mp4")
ret, frame = cali_video.read()
h, w = frame.shape[:2]
square_size = 2.4
pattern_size = (7, 7)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
obj_points =[]
img_points = []
img_names = []
index=0
figsize = (20, 20)
while(ret):
    index +=1
    if index%50 == 0:
        frame_cal_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(frame_cal_gray, pattern_size)
        if found:
            img_w_corners = cv2.drawChessboardCorners(frame, pattern_size, corners, found)
            img_w_corners =cv2.cvtColor (img_w_corners, cv2.COLOR_BGR2RGB)
            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)
            img_names.append(frame)
    

    ret, frame = cali_video.read()
cali_video.release()
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
renderer = mesh_renderer.MeshRenderer(camera_matrix, frame_width,frame_height,"drill\drill.obj")
# ========== run on all frames
while True:
   # cv2.resizeWindow('frame', 960, 540)
    ret, frame = cap.read()
 
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    farame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_farme, desc_frame = feature_extractor.detectAndCompute(farame_gray, None)
    kp_img_template, desc_img_template = feature_extractor.detectAndCompute(img_template_gray, None)
    test = cv2.drawKeypoints(img_template_rgb, kp_img_template, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(test)
    # plt.show()
    
    matches = bf.knnMatch(desc_frame, desc_img_template, k=2)
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance/m[1].distance < 0.5:
         good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]


    im_matches = cv2.drawMatchesKnn(frame_rgb, kp_farme, img_template_rgb, kp_img_template, good_and_second_good_match_list[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    good_kp_template = np.array([kp_img_template[m.trainIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([kp_farme[m.queryIdx].pt for m in good_match_arr])

    H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)

    # ++++++++ do warping of another image on template image
    # we saw this in SIFT notebook
    imgBGR = frame
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    fr_kp = good_kp_frame[masked.ravel()==1 ]
    tm_kp = good_kp_template[masked.ravel()==1]
   
    dim =2
   
    obj_width_cm = 25
    obj_height_cm = 15
    tm_kp_cm =  tm_kp / dim * (25, 15)
    tm_kp_cm_3d = np.column_stack((tm_kp_cm, np.zeros(tm_kp_cm.shape[0])))
    object_points = np.array(tm_kp_cm_3d)
    imagePoints = np.array(fr_kp)
    ret_val, r_vec, t_vec = cv2.solvePnP(object_points, imagePoints, camera_matrix, dist_coefs)

    
    drawn_image = renderer.draw(frame,r_vec, t_vec)
    dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)
    imgpts = cv2.projectPoints(objectPoints,r_vec, t_vec/300, camera_matrix, dist_coefs)[0]
    drawn_image1 = draw(dst, imgpts)
    drawn_image = cv2.cvtColor(drawn_image1, cv2.COLOR_BGR2RGB)
    # plt.imshow(drawn_image1)
    # plt.show()

    # =========== plot and save frame
   
    out.write(drawn_image)
    

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

# ======== end all
cap.release()
out.release()
cv2.destroyAllWindows()