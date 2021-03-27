import skimage
import scipy
import skimage.io as io
import matplotlib.pyplot as plt

# read data
avengers = io.imread('data/image_data/avengers_imdb.jpg')
bush = io.imread('data/image_data/bush_house_wikipedia.jpg')
forestry = io.imread('data/image_data/forestry_commission_gov_uk.jpg')
rolland = io.imread('data/image_data/rolland_garros_tv5monde.jpg')

#####################################
# Question 1
#####################################
# print image shape
print(avengers.size)

# grayscale and binaryscale
from skimage import color, filters

avengers_gray = color.rgb2gray(avengers)
plt.axis('off')
plt.imshow(avengers_gray, cmap='gray')
plt.savefig('outputs/avengers_gray.jpg')

avengers_binary = avengers_gray > filters.threshold_otsu(avengers_gray)
plt.imshow(avengers_binary, cmap='gray')
plt.savefig('outputs/avengers_binary.jpg')

#####################################
# Question 2
#####################################
# Add gaussian random noise
from skimage import util

bush_gaussian = util.random_noise(bush, mode='gaussian', var=0.1)
plt.imshow(bush_gaussian)
plt.savefig('outputs/bush_gaussian.jpg')

# Filter the perturbed image with a Gaussian mask (sigma=1)
# and a uniform smoothing mask(latter=9x9)
from scipy import ndimage
bush_gaussian_filter = ndimage.gaussian_filter(bush_gaussian, sigma=1)
plt.imshow(bush_gaussian_filter)
plt.savefig('outputs/bush_gaussian_filter.jpg')

bush_uniform_filter = ndimage.uniform_filter(bush_gaussian, size=9)
plt.imshow(bush_uniform_filter)
plt.savefig('outputs/bush_uniform_filter.jpg')

#####################################
# Question 3
#####################################
from skimage import segmentation
forestry_kmeans = segmentation.slic(forestry, n_segments=5, compactness=18,
                                    start_label=1, enforce_connectivity=True, convert2lab=True)
plt.imshow(forestry_kmeans)
plt.savefig('outputs/forestry_kmeans.jpg')

#####################################
# Question 4
#####################################
from skimage import feature, transform

# Perform Canny edge detection(canny is for grayscale image)
rolland_gray = color.rgb2gray(rolland)
rolland_edges = feature.canny(rolland_gray)
plt.imshow(rolland_edges, cmap=plt.cm.gray)
plt.savefig('outputs/rolland_edges.jpg')

# Apply Hough-Transform
import numpy as np
edges = transform.rotate(rolland_edges, 180)
edges = np.fliplr(edges)
rolland_hough = transform.probabilistic_hough_line(edges, threshold=160, line_length=110, line_gap=7)
plt.figure()
plt.axis('off')
for line in rolland_hough:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
plt.savefig('outputs/rolland_hough.jpg')
