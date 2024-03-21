#%%
import cv2 as cv
import numpy as np
'''
The script reads video from the laptop camera. I tested it on different recoreded videos but background has to be somewhat stable.
'''
def count_fingers(contour, defects):
    if defects is None:
        return 0

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        
        # Calculate the euclidean distance for each axis of the triangle
        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        # Calculate the angle between each finger. Ignore the angles that more than 90 degrees.
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
        if angle <= np.pi / 2:
            finger_count += 1

    return finger_count

def main():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Define the range of skin color in HSV
        lower_skin = np.array([0, 66, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv.inRange(hsv, lower_skin, upper_skin)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv.dilate(skin_mask, kernel, iterations=2)


        contours, hierarchy = cv.findContours(skin_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv.contourArea)

            # Ensure the contour is large enough. Otherwise, it would freeze and it is annoying
            if cv.contourArea(max_contour) > 1000:
                hull = cv.convexHull(max_contour)
                cv.drawContours(frame, [max_contour], -1, (0, 0, 255), 2) # contour

                cv.drawContours(frame, [hull], -1, (255, 0, 0), 2) # hull 
                hull_indices = cv.convexHull(max_contour, returnPoints=False)
                defects = cv.convexityDefects(max_contour, hull_indices)

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])
                        cv.line(frame, start, end, (255, 150, 0), 5)
                        cv.circle(frame, far, 5, (0, 255, 255), -1)
                fingers = count_fingers(max_contour, defects)

                # Print the number of fingers
                cv.putText(frame, f'Fingers: {int(fingers) + 1}',
                            (10, 50),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                            2)
                
        cv.imshow('Original', frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.waitKey(1) & 0xFF == ord('s'):
            cv.imwrite('./imges/saved_img_cont6.png', frame)
            print('saved!!')


    # Release the video capture object and close all windows
    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(1)

if __name__ == '__main__':
    main()

# %%
