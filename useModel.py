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
import Unet_old

imageNo = "227"

modelPath = "./results/Part1/32Images/model_2023-11-27094530.pt"
predictionLocation = "./images/IMPOSSIBLE_1_output.png"
IMG_HEIGHT = 944
IMG_WIDTH = 944


imageLabel = "slice__" + imageNo + ".tif"
#imagePath = "./data/data/SOCprist0" + imageNo + ".tiff"
imagePath = "./images/1IMPOSSIBLE.bmp"

# Setup Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Used: " + str(device))


def dice_loss(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    total_sum = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2);
    
    dice = 2.0*intersection/total_sum

    return (1 - dice).mean()

def calc_loss(pred, target, metrics, criterion, bce_weight=0.5):
    cce = criterion(pred, target)

    SoftMaxFunc = torch.nn.Softmax2d()
    pred = SoftMaxFunc(pred)

    dice = dice_loss(pred, target)
    loss = cce * bce_weight + dice * (1 - bce_weight)

    metrics['cce'] += cce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


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
model = Unet_old.UNet(3).to(device)
model.load_state_dict(torch.load(modelPath))
model.eval() 

# Make Prediction
pred_raw = model(input)
pred = torch.softmax(pred_raw, dim=1)
pred = pred.data.cpu().numpy()
print(pred.shape)

# Combine second and third channels for difficult images
combined_channels = np.maximum(pred[:, 1, :, :], pred[:, 2, :, :])

# Replace gray pixels with white
combined_channels[combined_channels > 0.5] = 1.0

# Set the second and third channels to the combined result
# pred[:, 1, :, :] = combined_channels
# pred[:, 2, :, :] = combined_channels

# Read True Label
blackmask = cv2.imread("./data/split_masks/black/" + imageLabel, 0) 
greymask  = cv2.imread("./data/split_masks/grey/"  + imageLabel, 0) 
whitemask = cv2.imread("./data/split_masks/white/" + imageLabel, 0) 

blackmask = transformers(blackmask)
greymask  = transformers(greymask)
whitemask = transformers(whitemask)

total_mask = torch.zeros((3, blackmask.size(1), blackmask.size(2)))
total_mask[0] = blackmask
total_mask[1] = greymask
total_mask[2] = whitemask

# Unsqueeze
total_mask = total_mask.unsqueeze(0)

# Calculate Loss
metrics = defaultdict(float)
criterion = torch.nn.CrossEntropyLoss()
loss = calc_loss(pred_raw.cpu(), total_mask, metrics, criterion)
print_metrics(metrics, 1, "val")


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
