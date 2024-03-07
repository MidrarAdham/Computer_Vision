#%%
import os
import cv2 as cv
import numpy as np
#%%
# Load a video using OpenCV:
cap = cv.VideoCapture('../sample_data/hands.MOV')
#%%
def process_frames():
    pass
#%%
# Read the video using OpenCV:
while True:
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    ret, frame = cap.read()
    if not ret:
        cv.destroyAllWindows()
        break
    ##
    process_frames()
    ##
    cv.imshow('Frame', frame)
    if cv.waitKey(20) & 0xff == 27:
        cv.destroyAllWindows()
        break
cap.release()