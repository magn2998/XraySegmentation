# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# THIS FILE IS USED TO SPLIT THE LABELED DATA INTO THREE MASKS #
# ------------ ONE MASK FOR EACH SEGMENTATION CLASS -----------#
# ------------------------------------------------------------ #


import numpy as np
from imutils import paths
import cv2
import os


# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images('./data/data')))
maskPaths = sorted(list(paths.list_images('./data/labels')))

# image_test = cv2.imread('./data/logo1.png', 0)
# print(image_test)

# Create Mask for Black Segments
for img_name in maskPaths:
    print(img_name)
    image = cv2.imread(img_name, 0)
    lower_bound = np.array([0])
    upper_bound = np.array([100])
    #masking the image using inRange() function
    imagemask = cv2.inRange(image, lower_bound, upper_bound)
    #displaying the resulting masked image
    cv2.imwrite("./data/split_masks/black/"+os.path.basename(img_name), imagemask)
    
# Create Masks for Grey Segments
for img_name in maskPaths:
    print(img_name)
    image = cv2.imread(img_name, 0)
    lower_bound = np.array([120])
    upper_bound = np.array([130])
    #masking the image using inRange() function
    imagemask = cv2.inRange(image, lower_bound, upper_bound)
    #displaying the resulting masked image
    cv2.imwrite("./data/split_masks/grey/"+os.path.basename(img_name), imagemask)
	
# Create Masks for White Segments
for img_name in maskPaths:
    print(img_name)
    image = cv2.imread(img_name, 0)
    lower_bound = np.array([200])
    upper_bound = np.array([256])
    #masking the image using inRange() function
    imagemask = cv2.inRange(image, lower_bound, upper_bound)
    #displaying the resulting masked image
    cv2.imwrite("./data/split_masks/white/"+os.path.basename(img_name), imagemask)
	
    



