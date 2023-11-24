import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
import Unet

imagePath = "./data/difficultData/Slice_ 1316.png"
modelPath = "./data/model_2023-11-24152752.pt"
predictionLocation = "./images/prediction_vs_original.png"
IMG_HEIGHT = 688
IMG_WIDTH = 688


# Setup Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Used: " + str(device))


# Read Image and setup input
def crop800(image):
    return crop(image, 0, 0, IMG_HEIGHT, IMG_WIDTH)

transformers = transforms.Compose([transforms.ToPILImage(),
    transforms.Lambda(crop800),
    transforms.ToTensor()])

image = cv2.imread(imagePath, 0) # Read as grayscaled 
image = transformers(image)
input = image.to(device)
input = input.unsqueeze(0) # Make the dimension match (c,H,W) -> (n, c, H, W)

print(input.shape)

# Setup Model
model = Unet.UNet(3).to(device)
model.load_state_dict(torch.load(modelPath))
model.eval() 

# Make Prediction
pred = model(input)
pred = torch.softmax(pred, dim=1)
pred = pred.data.cpu().numpy()
print(pred.shape)

# Combine second and third channels
combined_channels = np.maximum(pred[:, 1, :, :], pred[:, 2, :, :])

# Replace gray pixels with white
combined_channels[combined_channels > 0.5] = 1.0

# Set the second and third channels to the combined result
pred[:, 1, :, :] = combined_channels
pred[:, 2, :, :] = combined_channels

# Save Result
img_to_plot = pred[0].transpose(1, 2, 0)
plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.imshow(image.squeeze().cpu().numpy(), cmap="gray")
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(img_to_plot, cmap="gray")
plt.title("Prediction")

plt.tight_layout()
plt.savefig(predictionLocation)