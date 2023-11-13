import matplotlib_terminal
import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchvision.transforms.functional import crop
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from sklearn.model_selection import train_test_split
from imutils import paths
import cv2
import sys
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
	
    



