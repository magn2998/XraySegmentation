import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms.functional import crop
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
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
import random

# Global Variables
batch_size = 5
IMG_HEIGHT = 320
IMG_WIDTH = 320
EPOCHS = 5
NUM_SAMPLES = 340

# Auxillary Global Variables (Used for random cropping - see crop_rnd)
crop_x = 0
crop_y = 0
crop_v = True # When creating figure, we want to always start at (0,0) - Makes comparison much easier

print("IMAGE HEIGHT: " + str(IMG_HEIGHT))
print("IMAGE WIDTH: " + str(IMG_WIDTH))
print("MAX EPOCHS: " + str(EPOCHS))
print("NUM SAMPLES: " + str(NUM_SAMPLES))

# To Crop Images - Placement on picture is random every time to increase dataset
def crop_rnd(image):
    if crop_v:
        return crop(image, crop_x, crop_y, IMG_HEIGHT, IMG_WIDTH)
    else:
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
            # Setup Global Variables used for cropping
            global crop_x, crop_y
            #  Compute Random Cropping
            actual_height = image.shape[0]
            actual_width = image.shape[1]
            crop_x = random.randint(0, actual_width  - IMG_WIDTH  - 1)
            crop_y = random.randint(0, actual_height - IMG_HEIGHT - 1)
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
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]


transforms = transforms.Compose([transforms.ToPILImage(),
    transforms.Lambda(crop_rnd),
    transforms.ToTensor()])

# Load datasets
train_set = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
    transforms=transforms)
test_set = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)

#Change size of training set for experiment
train_set = Subset(train_set, range(NUM_SAMPLES))



train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)


def get_data_loaders():
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }

    return dataloaders


def dice_loss(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    total_sum = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2);
    
    dice = 2.0*intersection/total_sum

    return (1 - dice).mean();

def IoU_loss(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection;
    
    IoU = intersection/union

    return (1 - IoU).mean();


def calc_loss(pred, target, metrics, criterion, bce_weight=0.5):
    cce = criterion(pred, target)

    SoftMaxFunc = nn.Softmax2d()
    pred = SoftMaxFunc(pred)

    dice = dice_loss(pred, target)
    # IoU = IoU_loss(pred, target)
    # print(IoU)

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


def train_model(model, optimizer, scheduler, num_epochs=25):
    dataloaders = get_data_loaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    criterion = nn.CrossEntropyLoss()

    # Early Stopping Variables
    counter = 0
    tolerance = 5

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs)) #Reformatted to make it more readable in console
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics, criterion)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step() # Update Learning Rate

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                counter = 0
            elif phase == 'val':
                # Implementation of Early Stopping
                # Criteria: If no improvement of validation was made in 4 epocs
                counter = counter + 1
                if counter > tolerance:
                    print("No improvements seen in " + str(tolerance) + " epochs. Initiating Early Stopping.")
                    print('Best val loss: {:4f}'.format(best_loss))
                    model.load_state_dict(best_model_wts) # Load and return best model
                    return model
            
            


        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def run(UNet):
    num_class = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device Used: " + str(device))

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=1.0)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCHS)


    # Finish up and plot results
    model.eval()  # Set model to the evaluation mode

    global crop_v
    crop_v = False # Start at (0,0) for cropping
    inputs, labels, img_names = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    SoftMaxFunc = nn.Softmax2d()
    pred = SoftMaxFunc(pred)
    pred = pred.data.cpu().numpy()

    difference = np.abs(labels.cpu().numpy() - pred)
    
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
    plt.savefig('./images/picture_training_' + filename + ".png")


    # Save the model!
    torch.save(model.state_dict(), "./data/model_" + filename + ".pt")



    # Save the model!
    torch.save(model.state_dict(), "./data/model_" + filename + ".pt")
