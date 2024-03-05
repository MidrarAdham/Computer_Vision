import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

GOLDEN_RATIO = 1.61803398875
FIGURE_WIDTH = 10
FIGURE_HEIGHT = FIGURE_WIDTH / GOLDEN_RATIO

def RandomlyTransformImage(image):
    # imageTransformed = image.copy()
    # transform = np.eye(3)
    # return imageTransformed, transform
    
    # User-Specified Parameters
    npr = 100  # Maximum number of pixels to randomly move inside as part of the warp
    nbr = 50  # Maximum change in brightness
    nba = 0.2 # Maximum percent change in pixel scale (alpha), 1.0 is no change

    # Define the points of the original image
    p1 = [0,0]
    p2 = [0, image.shape[1]-1]
    p3 = [image.shape[0]-1, image.shape[1]-1]
    p4 = [image.shape[0]-1, 0]

    # Calculate the transformed points such that they are within the image
    q1 = np.random.randint(0,npr,2) + p1
    q2 = np.random.randint(0,npr,2) + [p2[0], p2[1] - npr]
    q3 = np.random.randint(0,npr,2) + p3 - npr
    q4 = np.random.randint(0,npr,2) + [p4[0] - npr, p4[1]]

    print(f'p1: {p1[0]:6.1f} {p1[1]:6.1f} q1: {q1[0]:4d} {q1[1]:4d}')
    print(f'p2: {p2[0]:6.1f} {p2[1]:6.1f} q2: {q2[0]:4d} {q2[1]:4d}')
    print(f'p3: {p3[0]:6.1f} {p3[1]:6.1f} q3: {q3[0]:4d} {q3[1]:4d}')
    print(f'p4: {p4[0]:6.1f} {p4[1]:6.1f} q4: {q4[0]:4d} {q4[1]:4d}')

    transform = cv.getPerspectiveTransform(np.float32([p1, p2, p3, p4]), np.float32([q1, q2, q3, q4]))
    imageTransformed = cv.warpPerspective(image, transform, (image.shape[1], image.shape[0]))

    # add a beta value to every pixel 
    brighten = np.random.randint(-nbr, nbr + 1)
    cv.add(imageTransformed, brighten, imageTransformed)               

    # multiply every pixel value by alpha
    scale = 1 + np.random.uniform(-nba, nba)
    cv.multiply(imageTransformed, scale, imageTransformed)
    
    return imageTransformed, transform

if __name__ == '__main__':

    image_filename = '../../sample_data/hw6_images_2/lauti.JPG'
    n_pixels_per_side = 500
    # n_pixels_per_side = 2000
    n_features = 50
    marker_size = 20

    imageBGR = cv.imread(image_filename)
    imageBGR = imageBGR[0:n_pixels_per_side, 0:n_pixels_per_side]
    image = cv.cvtColor(imageBGR, cv.COLOR_BGR2RGB)
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    sift = cv.SIFT_create()
    sift.setNFeatures(n_features)
    kp = sift.detect(image, None)
    kp = sorted(kp, key=lambda x: -x.response)
    kp = kp[:n_features]

    imageTransformed, transform = RandomlyTransformImage(image)
    
    sift2 = cv.SIFT_create()
    sift2.setNFeatures(n_features)
    kpt = sift2.detect(imageTransformed, None)
    kpt = sorted(kpt, key=lambda x: -x.response)
    kpt = kpt[:n_features]
    
    print(f'Number of keypoints            : {len(kp)}')
    print(f'Number of keypoints transformed: {len(kpt)}')

    # Display the images side-by-side
    figure = plt.figure(figsize=(FIGURE_WIDTH*2, FIGURE_WIDTH))

    axes = figure.add_subplot(1, 2, 1)
    axes.imshow(image, cmap='gray')
    for k in kp:
        x, y = k.pt
        axes.plot(x, y, 'r', marker='.', alpha=0.8, markersize=marker_size, markeredgewidth=0)
    
    axes = figure.add_subplot(1, 2, 2)
    axes.imshow(imageTransformed, cmap='gray')
    
    matches = np.zeros((n_features), dtype=bool)
    for k in kp:
        x, y = k.pt
        print(x,'\t', y)
        xn, yn, l = transform @ np.array([x, y, 1])
        # print(xn,'\t',yn, '\t', l)
        xt = xn/l
        yt = yn/l
        
        # Determine how far the nearest keypoint is from the transformed keypoint
        for c,k2 in enumerate(kpt):
            if matches[c] == True: # If this keypoint has already been matched,
                continue
            x2, y2 = k2.pt
            distance = np.sqrt((x2 - xt)**2 + (y2 - yt)**2)
            if distance<3:
                matches[c] = True
        
        axes.plot(xt, yt, 'r', marker='.', alpha=0.8, markersize=marker_size, markeredgewidth=0)
    for c,k2 in enumerate(kpt):
        x, y = k2.pt
        #axes.plot(x, y, 'g', marker='.', alpha=0.8)
        if matches[c] == True:
            axes.plot(x, y, 'b', marker='.', alpha=0.8, markersize=marker_size, markeredgewidth=0)
        else:
            axes.plot(x, y, 'y', marker='.', alpha=0.8, markersize=marker_size, markeredgewidth=0)       
    axes.set_title(f'Percent of matches: {np.sum(matches)/n_features*100:4.1f}%')

    plt.show()