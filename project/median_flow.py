#%%
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#%%
# filename = 'hands_stable.mov'
# filename = 'hands_stable_bg.mov'
# filename = 'hands_test.mov'
filename = 'hands_test2.mov'
cap = cv.VideoCapture(filename)

def extract_background(frame):
    backsub = cv.createBackgroundSubtractorKNN()
    return backsub.apply(frame)

def detect_skin(frame):
    # Convert BGR to YCbCr
    ycbcr = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    
    # Skin color range in YCbCr
    lower = np.array([0, 138, 67], dtype=np.uint8)
    upper = np.array([255, 173, 133], dtype=np.uint8)
    
    # Mask the skin color
    mask = cv.inRange(ycbcr, lower, upper)
    
    # Apply morphological operations to remove noise
    # mask = cv.medianBlur(mask, 5)
    mask = cv.GaussianBlur(mask, (5,5), 10)
    mask = cv.dilate(mask, None, iterations=2)
    mask = cv.erode(mask, None, iterations=2)

    
    return mask

# Capture video from webcam
backsub = cv.createBackgroundSubtractorKNN()

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv.resize(frame.copy(), (640, 800))
    fg_frame = backsub.apply(frame)
    # cv.imshow('Background', fg_frame)
    
    skin_mask = detect_skin(frame)
    cv.imshow('Skin Mask', skin_mask)

    contours, _ = cv.findContours(skin_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        convex_hull = cv.convexHull(contours[0])
        cv.drawContours(frame, [convex_hull], -1, (150, 155, 0), 2)
        convexityDefects = cv.convexityDefects(contours[0],convex_hull)
    


    # Display the frame
    cv.imshow('Hand Tracking', frame)

    # cv.imshow('Skin Mask', convexityDefects)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()