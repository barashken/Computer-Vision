# Written by:
# Aviv Galily 316431774
# Bar Ashkenazi 313233181

# Import necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Remove irrelevant segments of the image and retain only the lane portion
def region_of_interest(img):
    mask = np.zeros_like(img)
    match_mask_color = 255

    height = img.shape[0]
    width = img.shape[1]

    polygons = np.array([[(int(width/4), height), (int(width/2), int(height/1.5) + 50), (int(width/4) + int(width/2) + 300, height)]]) 
    
    cv2.fillPoly(mask, polygons, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Returns coordinates of the line
def get_coordinates(image, params):
    slope = params[0]
    intercept = params[1]
    
    y1 = image.shape[0]     
    y2 = int(y1 * (2/5)) 
    x1 = int((y1 - intercept) / slope) 
    x2 = int((y2 - intercept) / slope) 
    
    return np.array([x1, y1, x2, y2])

# Returns averaged lines on left and right sides of the image
# Average the Hough lines as left or right lanes
def avg_lines(image, lines, averaged_lines): 
    left = [] 
    right = [] 
    
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4)
          
        # Fit polynomial, find intercept and slope
        if y1 != y2 and x1 != x2:
            params = np.polyfit((x1, x2), (y1, y2), 1)  
            slope = params[0] 
            y_intercept = params[1] 
    
            if -0.9 < slope < -0.5: 
                left.append((slope, y_intercept)) #Negative slope = left lane
            elif 0.9 > slope > 0.5: 
                right.append((slope, y_intercept)) #Positive slope = right lane
    
    # Avg over all values for a single slope and y-intercept value for each line
    # Find x1, y1, x2, y2 coordinates for left & right lines
    if len(left) != 0:  
        left_avg = np.average(left, axis = 0) 
        left_line = get_coordinates(image, left_avg)
    else:
        left_line = averaged_lines[0]

    if len(right) != 0:
        right_avg = np.average(right, axis = 0) 
        right_line = get_coordinates(image, right_avg)
    else:
        right_line = averaged_lines[1]

    left, right = find_if_turning(left_line, right_line)
    return find_connectd_points(left_line, right_line), left, right

def find_if_turning(left_line, right_line):
    m_left = (left_line[3] - left_line[1]) / (left_line[2] - left_line[0])
    m_right = (right_line[3] - right_line[1]) / (right_line[2] - right_line[0])
    if (m_right < 0.5006 and -0.68 < m_left < -0.66):
        return False, True
    if (-0.5 < m_left and 0.66 < m_right < 0.7):
        return True, False
    return False, False
 
def find_connectd_points(left_line, right_line):
    m_left = (left_line[3] - left_line[1]) / (left_line[2] - left_line[0])
    m_right = (right_line[3] - right_line[1]) / (right_line[2] - right_line[0])
    b_left = left_line[1] - m_left * left_line[0]
    b_right = right_line[1] - m_right * right_line[0]
    x_intersect = (b_right - b_left) / (m_left - m_right)
    y_intersect = m_left * x_intersect + b_left
    return np.array([left_line[0], left_line[1], int(x_intersect), int(y_intersect)]), np.array([right_line[0], right_line[1],int(x_intersect), int(y_intersect)])

# Draw the lines on the frame
def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    x1,y1,x2,y2 = lines[0]
    x3,y3,x4,y4 = lines[1] 

    if (x1 != x2 and y1 != y2) and (x3 != x4 and y3 != y4):
        points = [(x1, y1), (x2 - 30, y2 + 30), (x4 + 30, y4 + 30), (x3, y3)] #[(0, 0), (0, img.shape[0]), (img.shape[1], img.shape[0]), (img.shape[1], 0)]]
        cv2.fillPoly(blank_image, np.array([points]), (0, 0, 255))
        # Draw the lines on top of the filled region
        cv2.line(blank_image, points[0], points[1], (0, 0, 255), thickness=10)
        cv2.line(blank_image, points[3], points[2], (0, 0, 255), thickness=10)

    return blank_image

# Hough transform params
rho = 2 
theta = np.pi / 180 
threshold = 50   
min_line_length = 5  
max_line_gap = 5  

averaged_lines = []
xy_vec = []
counter_R = 0
counter_L = 0

# Read the video
cap = cv2.VideoCapture('input.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#cv2.WINDOW_NORMAL makes the output window resizealbe
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while(cap.isOpened()):
    # Read the frame
    ret, frame = cap.read()
    if ret == False:
        break

    height = frame.shape[0]
    width = frame.shape[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray, 50, 150)
    cropped_image = region_of_interest(canny_edges)
    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    averaged_lines ,left ,right = avg_lines(cropped_image, lines, averaged_lines)  
    image_with_lines = draw_the_lines(frame, averaged_lines)
    lanes = cv2.addWeighted(frame,1, image_with_lines, 0.5, 0)

    if(left or counter_L > 0):
        if(counter_L == 0):
            counter_L = 120

        counter_L = counter_L - 1
        cv2.putText(img=lanes, text="Move Left", org=(int(width / 4), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3.0, color=(0, 0, 255),thickness=6)

    
    if(right or counter_R > 0):
        if(counter_R == 0):
            counter_R = 120
        
        counter_R = counter_R - 1
        cv2.putText(img=lanes, text="Move Right", org=(width - int(width / 4) - 100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3.0, color=(0, 0, 255),thickness=6)

    #resize the window
    cv2.resizeWindow('frame', 960, 540)
    # Show the frame and save him
    out.write(lanes)
    cv2.imshow('frame', lanes)

    # Wait for the user to press a key
    if cv2.waitKey(1) == 27 & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
out.release()
cv2.destroyAllWindows()