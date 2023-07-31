import gdown
import os
from glob import glob
import shutil
import cv2
import numpy as np
import math

def set_far_one_to_zero(matrix):
    # Find the center point of all 1s in the matrix.
    center_x, center_y = 0, 0
    count_ones = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                center_x += i
                center_y += j
                count_ones += 1

    center_x = center_x // count_ones
    center_y = center_y // count_ones

    # Calculate the distance of each pixel from the center point.
    distances = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                distance = math.sqrt((i - center_x)**2 + (j - center_y)**2)
                distances.append(distance)

    median = np.median(distances)

    # Iterate over all the pixels in the matrix and set the value to 0 if the pixel is far enough from the center of all 1s.
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                distance = math.sqrt((i - center_x)**2 + (j - center_y)**2)
                if distance > median * 2.8:
                    matrix[i][j] = 0

    return matrix

def crop_img(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask
    ret, mask = cv2.threshold(gray, 254, 1, cv2.THRESH_BINARY_INV)

    mask = set_far_one_to_zero(mask)

    mask = mask * 255

    # Find the coordinates of the non-background pixels
    y_pixels, x_pixels = np.nonzero(mask)

    # Find the most extreme positions of the non-background pixels in each direction
    x_min = np.min(x_pixels)
    x_max = np.max(x_pixels)
    y_min = np.min(y_pixels)
    y_max = np.max(y_pixels)

    return img[y_min:y_max+1, x_min:x_max+1], (x_min, x_max, y_min, y_max)

to_route = 'clean_dataset'
from_route = 'unzip/SketchQuery_Test'

try:
    shutil.rmtree(to_route)
except:
    pass

try:
    os.mkdir(to_route)
except:
    pass

for img_path in glob(from_route + '/*'):
    print(img_path)

    img_to = os.path.join(to_route, f'{os.path.basename(img_path)}')
    img_to = img_to.replace('.jpg', '.png')

    img_read = cv2.imread(img_path)
    
    crop, bbox = crop_img(img_read)
    cv2.imwrite(img_to, crop)





