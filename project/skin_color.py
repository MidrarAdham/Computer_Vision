#%%
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%%
# filename = '../sample_data/hands.MOV'
filename = 'hands_stable_bg.mov'
cap = cv.VideoCapture(filename)

min_ycrcb = np.array([0,135, 150],np.uint8)
max_ycrcb = np.array([255,180,135],np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img_ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCR_CB)
    mask_ycrcb  = cv.inRange(img_ycrcb, min_ycrcb, max_ycrcb)
    mask_ycrcb = cv.morphologyEx(mask_ycrcb, cv.MORPH_OPEN, np.ones((3,3),np.uint8))

    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # gaus_img = cv.GaussianBlur(img_hsv, (21,21), 0)
    mask_hsv = cv.inRange(img_hsv, (0,15,0),(17,170,255))
    mask_hsv = cv.morphologyEx(mask_hsv, cv.MORPH_OPEN, np.ones((3,3),np.uint8))
    
    global_mask = cv.bitwise_and(mask_ycrcb, mask_hsv)
    global_mask = cv.medianBlur(global_mask,3)
    global_mask = cv.morphologyEx(global_mask, cv.MORPH_CLOSE, np.ones((5,5),np.uint8))

    hsv_results = cv.bitwise_not(mask_hsv)
    hsv_results = cv.threshold(hsv_results, 0, 150, cv.ADAPTIVE_THRESH_MEAN_C)[1]

    ycr_results = cv.bitwise_not(mask_ycrcb)
    global_results = cv.bitwise_not(global_mask)





    # frame_color = cv.cvtColor(frame, cv.COLOR_BGR2YCR_CB)
    # skin = cv.inRange(frame_color, min_ycrcb, max_ycrcb)
    # skin_frame = cv.bitwise_and(frame, frame, mask = skin)
    cv.imshow('frame', frame)
    cv.imshow('hsv', hsv_results)
    # cv.imshow('YCRBC', ycr_results)
    # cv.imshow('Global Results', global_results)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()