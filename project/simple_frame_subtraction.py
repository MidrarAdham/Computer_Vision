#%%
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%%
# filename = '../sample_data/hands.MOV'
filename = './hands_stable_bg.mov'
cap = cv.VideoCapture(filename)
first_frame = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_diff = cv.absdiff(first_frame,frame)
    first_frame = frame.copy()
    cv.imshow('frame', frame_diff)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()