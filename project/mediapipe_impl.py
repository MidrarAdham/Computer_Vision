#%%
import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
#%%
# Initialize variables:
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#%%
# Load a video using OpenCV:
cap = cv.VideoCapture('../sample_data/hands.MOV')
#%%
def media_pip_hands_tracking(frame):
    # Convert the image to RGB:
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Make the detection:
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                return frame
#%%
def process_frames(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(frame, 100, 200)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    hull = cv.convexHull(largest_contour)
    mask = np.zeros_like(frame)
    cv.drawContours(mask, [hull], -1, 255, -1)
    frame = cv.bitwise_and(frame, mask)
    return frame

def hands_detection(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(frame, 100, 200)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    hull = cv.convexHull(largest_contour)
    mask = np.zeros_like(frame)
    cv.drawContours(mask, [hull], -1, 255, -1)
    frame = cv.bitwise_and(frame, mask)
    return frame
    # return edges

def background_subtraction (frame):
    backsub = cv.createBackgroundSubtractorMOG2()
    foreground_mask = backsub.apply(frame)
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return frame, foreground_mask

#%%

img = cv.imread('../sample_data/hand.jpg')
img = hands_detection(img)
plt.imshow(img, cmap='gray')

#%%
# Read the video using OpenCV:
while True:
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    ret, frame = cap.read()
    frame = cv.resize(frame, (600, 800))
    if not ret:
        cv.destroyAllWindows()
        break
    ##
    # frame = process_frames(frame)
    # frame, fgmask = background_subtraction(frame)
    frame = media_pip_hands_tracking(frame)
    ##
    cv.imshow('Frame', frame)
    # cv.imshow('Edges', edges)
    # cv.imshow('whatever', fgmask)
    if cv.waitKey(60) & 0xff == 27:
        cv.destroyAllWindows()
        break
cap.release()