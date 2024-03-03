import os
import random
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def show_images (dir, image):
    _, ax = plt.subplots(2, 5, figsize=(15, 5))
    ax = ax.flatten()
    
    for img, ax in zip(image, ax):
        img = cv.imread(dir + img, cv.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')

    plt.tight_layout()
    plt.show()


def rotate_images(dir, img):
    
    image = cv.imread(dir + img, cv.IMREAD_GRAYSCALE)
    std_dev = 1
    k = 150
    pos_1 = np.random.randint(0, k, 2)
    pos_2 = np.random.randint(0, k, 2)
    pos_3 = np.random.randint(0, k, 2)
    pos_4 = np.random.randint(0, k, 2)
    angle = random.randint(-10, 10)

    # Initialize the rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(center = (image.shape[1] / 2, image.shape[0] / 2), angle=angle, scale=1)
    # Add a third row for the rotation matrix so considering homogensous coordinates
    rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])
    # Apply rotation to image
    rotated_image = cv.warpPerspective(image, rotation_matrix, (image.shape[1], image.shape[0]))
    # Define the image corners after rotation (Not really doing what it's supposed to do, change later if there is time)
    corner_1 = np.float32([0,0])
    corner_2 = np.float32([0, rotated_image.shape[1]-1])
    corner_3 = np.float32([rotated_image.shape[0]-1,rotated_image.shape[1]-1])
    corner_4 = np.float32([rotated_image.shape[0]-1, 0])

    # Define the source and destination points (Similar to Dr. McNames code)
    src_points = np.float32([corner_1, corner_2, corner_3, corner_4])
    dst_points = np.float32([corner_1 + pos_1, corner_2 + [pos_2[0], pos_2[1] - k], corner_3 + pos_3 - k, corner_4 + [pos_4[0] - k, pos_4[1]]])

    # Get the projective matrix
    projective_matrix = cv.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv.warpPerspective(image, projective_matrix, (image.shape[1], image.shape[0]))

    return image, warped_image

def harris_corner(img):
    # dst = cv.cornerHarris(img=img, blockSize = 2, ksize = 3, k = 0.04)
    # img = np.float32(img)
    dst = cv.cornerHarris(src=img, blockSize=2, ksize=3, k=0.04)
    dst = cv.dilate(img, None)
    # img[dst > 0.01 * dst.max()] = 255
    # img = np.uint8(img)
    
    return dst

if __name__ == '__main__':
    dir = '../../sample_data/hw6_images_2/'
    orig, rotated = rotate_images(dir, 'lauti.JPG')
    dst = harris_corner(img=orig)
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.imshow(dst, cmap='gray')
    plt.show()
    
    # images = os.listdir(dir)
    # show_images(dir=dir, image=images)
    