# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib_inline
# %%

x = np.random.randint(2,4,14)
img = cv.imread('../../sample_data/new/lauti.JPG', cv.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
