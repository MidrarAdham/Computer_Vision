#%%
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%%
def detect_skin(frame):
    # Convert BGR to YCbCr
    ycbcr = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    
    # Skin color range in YCbCr
    lower = np.array([0, 138, 67], dtype=np.uint8)
    upper = np.array([255, 173, 133], dtype=np.uint8)
    
    # Mask the skin color
    mask = cv.inRange(ycbcr, lower, upper)
    
    # Apply morphological operations to remove noise
    mask = cv.medianBlur(mask, 5)
    mask = cv.dilate(mask, None, iterations=2)
    mask = cv.erode(mask, None, iterations=2)
    
    return mask

filename = 'hands_stable_bg.mov'
# Capture video from webcam
cap = cv.VideoCapture(filename)

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
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv.Canny(gray, 100, 200)

        # Find contours in the edge-detected image
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Find the convex hull of the largest contour
        hull = cv.convexHull(hand_contour, returnPoints=False)

        # Find convexity defects to identify fingertips
        defects = cv.convexityDefects(hand_contour, hull)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])

                # Draw circles at the fingertips
                cv.circle(frame, far, 5, (0, 0, 255), -1)

    # Display the frame
    cv.imshow('Hand Tracking', frame)

    # Check for 'q' key to quit
    if cv.waitKey(50) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv.destroyAllWindows()
