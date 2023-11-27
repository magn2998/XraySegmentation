import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import crop
import torch
import torch.nn as nn
from collections import defaultdict
import cv2
import Unet
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from imutils import paths
import time
import os

### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###
### THIS FILE IS USED TO TEST AND A VALIDATION SET, AS WHEN TRAINING ###
### ---------------------------------------------------------------- ###
### ---------------------------------------------------------------- ###

BATCH_SIZE = 5
SEGMENTS_WIDTH   = 32 # Height of individual segments, which are cropped section of the original image
SEGMENTS_HEIGHT  = 32 # Same as above, just for height
SEGMENTS_OVERLAP = 10  # Pixels to overlap between segments

modelPath = "./results/Part1/32Images/model_2023-11-27094530.pt" # Model to use




# Setup Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device Used: " + str(device))


# Crop Image
def crop(img, start_x, start_y, width, height):
    return img[start_y:start_y+height, start_x:start_x+width]

# Merge segment sections back together
def merge_segment(full_img, segment, start_x, start_y, end_x, end_y, segment_x, segment_y):
    temp_start_y = start_y
    temp_segment_y = segment_y
    while start_x < end_x:
        while start_y < end_y:
            full_img[0][start_y][start_x] = segment[0][segment_y][segment_x]
            full_img[1][start_y][start_x] = segment[1][segment_y][segment_x]
            full_img[2][start_y][start_x] = segment[2][segment_y][segment_x]

            start_y += 1
            segment_y += 1
        start_y = temp_start_y
        segment_y = temp_segment_y
        start_x += 1 
        segment_x += 1   

# Merge all segments together to a single image
def merge_all_segments(img_arr, segments, seg_size, min_overlap):
    H, W = img_arr.shape
    Ws, Hs = seg_size

    # Create Full Array
    full_arr = np.zeros((3, W, H))

    # Merge the segments
    idx_x = 0
    i = 0
    while i < W:
        o = 0
        idx_y = 0
        while o < H:
            start_x = int(max(i - min_overlap/2, 0))
            start_y = int(max(o - min_overlap/2, 0))

            end_x = int(min(max(i - min_overlap, 0)+Ws-min_overlap/2, W))
            end_y = int(min(max(o - min_overlap, 0)+Hs-min_overlap/2, H))

            # Edge case - If last segment just goes to edge, just take entire thing
            if max(i - min_overlap, 0)+Ws == W:
                end_x = W
            if max(o - min_overlap, 0)+Hs == H:
                end_y = H


            # If segments overflows, make segment so it covers rest while maintaining size
            if start_x + Ws >= W:
                start_x = int(W - Ws + min_overlap/2)
            if start_y + Hs >= H:
                start_y = int(H - Hs + min_overlap/2)

            # Now that we've computed the placements inside the full picture, compute the indexes from the segments
            segment_x = max(i - min_overlap, 0)
            segment_y = max(o - min_overlap, 0)

            # If segments overflows, make segment so it covers rest while maintaining size
            if segment_x + Ws >= W:
                segment_x = W - Ws
            if segment_y + Hs >= H:
                segment_y = H - Hs

            segment_x = start_x - segment_x
            segment_y = start_y - segment_y

            merge_segment(full_arr, segments[idx_x][idx_y], start_x, start_y, end_x, end_y, segment_x, segment_y)
            
            idx_y += 1
            o = max(o - min_overlap, 0) + Hs

        idx_x += 1
        i = max(i - min_overlap, 0) + Ws

    return torch.from_numpy(full_arr)



# Create Image Segmentations
def create_segments(img_arr, seg_size, min_overlap):
    H, W = img_arr.shape
    Ws, Hs = seg_size
    O = min_overlap

    # Segment the array
    segments = []

    idx_x = 0
    i = 0
    while i < W:
        o = 0
        idx_y = 0
        segments.append([])

        while o < H:
            segments[idx_x].append([np.zeros((Ws, Hs))])

            segment_x = max(i - min_overlap, 0)
            segment_y = max(o - min_overlap, 0)

            # If segments overflows, make segment so it covers rest while maintaining size
            if segment_x + Ws >= W:
                segment_x = W - Ws
            if segment_y + Hs >= H:
                segment_y = H - Hs

            segments[idx_x][idx_y] = crop(img_arr, segment_x, segment_y, Ws, Hs)
            
            
            idx_y += 1
            o = segment_y + Hs

        
        idx_x += 1
        i = segment_x + Ws

    return segments

def dice_loss(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    total_sum = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2);
    
    dice = 2.0*intersection/total_sum

    return (1 - dice).mean()

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
        total_mask = torch.zeros((3, image.size(1), image.size(2)))
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
    transforms.ToTensor()])

# Load datasets
test_set = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

inputs, labels, img_names = next(iter(test_loader))

# Setup Model
model = Unet.UNet(3).to(device)
model.load_state_dict(torch.load(modelPath))
model.eval() 


# Split image into smaller overlapping images
merged_inputs = []

for input in inputs:
    print("Analysing next picture!")
    input = input.squeeze(0)
    segments = create_segments(input, (SEGMENTS_WIDTH, SEGMENTS_HEIGHT), SEGMENTS_OVERLAP)
    # Make Prediction
    for i in range(len(segments)):
        for o in range(len(segments[i])):
            model_input = segments[i][o].unsqueeze(0).unsqueeze(0).to(device)
            pred = model(model_input)
            pred = pred.data.cpu().numpy()
            segments[i][o] = pred.squeeze(0)

    # Combine Images
    merged = merge_all_segments(input, segments, (SEGMENTS_WIDTH, SEGMENTS_HEIGHT), SEGMENTS_OVERLAP)
    merged_inputs.append(merged)

# Create tensor
merged_inputs = torch.stack(merged_inputs)

# Calculate Loss
metrics = defaultdict(float)
criterion = nn.CrossEntropyLoss()
loss = calc_loss(merged_inputs, labels, metrics, criterion)
print_metrics(metrics, BATCH_SIZE, "val")

# Normalize the difference values to be in the range [0, 1]
SoftMaxFunc = nn.Softmax2d()
merged_inputs = SoftMaxFunc(merged_inputs)

difference = np.abs(labels.cpu().numpy() - merged_inputs.cpu().numpy())


images = [inputs.cpu().numpy(),  labels.cpu().numpy(), merged_inputs.cpu().numpy(), difference]

nrow = merged_inputs.shape[0]
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
plt.savefig('./images/picture_batchtestSegment_' + filename + ".png")
