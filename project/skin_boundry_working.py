#%%
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#%%

filename = 'hands_stable_bg.mov'
cap = cv.VideoCapture(filename)

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
    mask = cv.GaussianBlur(mask, (27, 27), 0)
    mask = cv.dilate(mask, None, iterations=2)
    mask = cv.erode(mask, None, iterations=2)
    
    return mask

# Capture video from webcam

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Detect skin in the frame
    skin_mask = detect_skin(frame)
    

    # Find contours in the skin mask
    contours, _ = cv.findContours(skin_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (hand)
    if len(contours) > 0:
        hand_contour = max(contours, key=cv.contourArea)

        # Draw bounding box around the hand
        x, y, w, h = cv.boundingRect(hand_contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (20, 100, 150), 2)

    # Display the frame
    cv.imshow('Hand Tracking', frame)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()