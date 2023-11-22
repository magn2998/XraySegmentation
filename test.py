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
import datetime

# Global Variables
batch_size = 5
IMG_HEIGHT = 400
IMG_WIDTH = 400
EPOCHS = 1

# To Crop Images
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

        image = cv2.imread(imagePath, 0) # Read as grayscaled 
        blackmask = cv2.imread("./data/split_masks/black/"+os.path.basename(maskPath), 0) 
        greymask  = cv2.imread("./data/split_masks/grey/" +os.path.basename(maskPath), 0) 
        whitemask = cv2.imread("./data/split_masks/white/"+os.path.basename(maskPath), 0) 

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            blackmask = self.transforms(blackmask)
            greymask = self.transforms(greymask)
            whitemask = self.transforms(whitemask)
        # return a tuple of the image and its mask
        
        total_mask = torch.zeros((3,IMG_HEIGHT,IMG_WIDTH))
        total_mask[0] = blackmask
        total_mask[1] = greymask
        total_mask[2] = whitemask

        # print()
        # print("Hello")
        # print(imagePath)
        # print(maskPath)
        # print(image)
        # print(image.shape)
        # print(total_mask.shape)
        # print(total_mask)
        # sys.exit()
        return (image, total_mask)

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
    transforms.Lambda(crop800),
    transforms.ToTensor()])

# Load datasets
train_set = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
    transforms=transforms)
test_set = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        for o in range(len(img_array[i])):
            imgArray = img_array[i][o].transpose(1,2,0) # Reshape to match (H,W,C) from (C,H,W)
            plots[i // ncol, i % ncol]
            plots[i // ncol, i % ncol].imshow(imgArray, cmap='gray')

    plt.savefig('./images/pictures_newBN.png')
    


def plot_side_by_side(img_arrays):
    # print(img_arrays)
    # flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))
    # print(flatten_list.shape)
    plot_img_array(img_arrays, ncol=len(img_arrays))





def get_data_loaders():
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }

    return dataloaders


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # pred = torch.sigmoid(pred)
    pred = torch.softmax(pred, dim=1)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
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

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

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

    model.eval()  # Set model to the evaluation mode

    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    # ])
    # # Create another simulation dataset for test
    # Get the first batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    # pred = torch.sigmoid(pred)
    pred = torch.softmax(pred, dim=1)
    pred = pred.data.cpu().numpy()
    print(pred.shape)

    # Change channel-order and make 3 channels for matplot
    # input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    # Map each channel (i.e. class) to each color
    # target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
    # pred_rgb = [masks_to_colorimg(x) for x in pred]

    # plot_side_by_side([inputs.cpu().numpy(),  labels.cpu().numpy(), pred])


    #Subtract images pred - labels 
    # Assuming labels and pred have the same shape
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
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.1)  # Add color bar to each subplot
                cbar.ax.tick_params(labelsize=8)  # Adjust font size of the color bar ticks
            else:
                ax.imshow(img.transpose(1, 2, 0), cmap="gray")  # Transpose the image dimensions from [channel, height, width] to [height, width, channel]

    plt.tight_layout()
    plt.show()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'./images/pictures_{timestamp}.png'
    plt.savefig(filename)

    # Save the model!
    torch.save(model.state_dict(), "./data/model.pt")