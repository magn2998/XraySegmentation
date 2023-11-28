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

### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
### THIS FILE IS USED TO TEST AND A VALIDATION SET, AS WHEN TRAINING ###
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###

BATCH_SIZE = 5
IMG_HEIGHT = 496
IMG_WIDTH = 496

modelPath = "./results/Part1/256Images/model_2023-11-28203144.pt"



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


# Thanks to: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]

        img_name = os.path.basename(imagePath)
        name = os.path.basename(maskPath)

        image = cv2.imread(imagePath, 0) # Read as grayscaled 
        blackmask = cv2.imread("./data/split_masks/black/"+name, 0) 
        greymask  = cv2.imread("./data/split_masks/grey/" +name, 0) 
        whitemask = cv2.imread("./data/split_masks/white/"+name, 0) 

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            blackmask = self.transforms(blackmask)
            greymask = self.transforms(greymask)
            whitemask = self.transforms(whitemask)
        
        # return a tuple of the image and its masks
        total_mask = torch.zeros((3,IMG_HEIGHT,IMG_WIDTH))
        total_mask[0] = blackmask
        total_mask[1] = greymask
        total_mask[2] = whitemask

        return (image, total_mask, img_name)

# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images('./data/data')))
maskPaths = sorted(list(paths.list_images('./data/labels')))
# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePaths, maskPaths,
    test_size=0.15, random_state=42)
# unpack the data split
(_, testImages) = split[:2]
(_, testMasks) = split[2:]


transforms = transforms.Compose([transforms.ToPILImage(),
    transforms.Lambda(crop800),
    transforms.ToTensor()])

# Load datasets
test_set = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)


# Setup Model
model = Unet_old.UNet(3).to(device)
model.load_state_dict(torch.load(modelPath))
model.eval() 

# Make Prediction
inputs, labels, img_names = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

# Predict
pred_raw = model(inputs)
SoftMaxFunc = torch.nn.Softmax2d()
pred = SoftMaxFunc(pred_raw)
pred = pred.data.cpu().numpy()
difference = np.abs(labels.cpu().numpy() - pred)

# Calculate Loss
metrics = defaultdict(float)
criterion = torch.nn.CrossEntropyLoss()
loss = calc_loss(pred_raw, labels, metrics, criterion)
print_metrics(metrics, BATCH_SIZE, "val")


# Normalize the difference values to be in the range [0, 1]
images = [inputs.cpu().numpy(),  labels.cpu().numpy(), pred, difference]

nrow = pred.shape[0]
ncol = len(images)
# Create a 5x3 subplot grid
fig, axs = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 5, nrow * 5)) # Adjust figsize as needed

for i in range(len(images)):  # For each of the 4 pictures
    for j in range(5):  # For each of the 5 versions
        ax = axs[j, i]
        img = images[i][j]  # Access the image; images should be a 3D array: [channel, height, width]
        if i == ncol - 1:  # Use a different colormap for the difference plot
            im = ax.imshow(img.transpose(1, 2, 0), cmap="viridis")
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.1)  # Add color bar to each subplot
        else:
            ax.imshow(img.transpose(1, 2, 0), cmap="gray")  # Transpose the image dimensions from [channel, height, width] to [height, width, channel]
            if i == ncol - 2:
                ax.text(0.5, -0.1, img_names[j], fontsize=12, ha='center', va='center', transform=ax.transAxes)
            
            
plt.tight_layout()
plt.show()
filename =  time.strftime("%Y-%m-%d%H%M%S")
plt.savefig('./images/picture_batchtest_' + filename + ".png")
