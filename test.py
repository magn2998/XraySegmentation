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

# Global Variables
batch_size = 5

# To Crop Images
def crop800(image):
    return crop(image, 0, 0, 192, 192)

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
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		# image = cv2.imread(imagePath)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# mask = cv2.imread(maskPath, 0)
		# mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        ## Test with Greyscaled
		image = cv2.imread(imagePath, 0)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		mask = cv2.imread(maskPath, 0)
		# mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2GRAY)

		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask

		# print("Hello")
		# print(imagePath)
		# print(maskPath)
		# print(image)
		# print(mask)
		# print(image.shape)
		# print(mask.shape)
		# sys.exit()
		return (image, mask)

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
 	# transforms.Resize((192, 192)),
    # transforms.functional.crop(top=0, left=0, height=192, width=192),
    transforms.Lambda(crop800),
	transforms.ToTensor()])

# Load datasets
train_set = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transforms)
test_set = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

# def generate_random_data(height, width, count):
#     x, y = zip(*[generate_img_and_mask(height, width) for i in range(0, count)])

#     X = np.asarray(x) * 255
#     X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
#     Y = np.asarray(y)

#     return X, Y


# def generate_img_and_mask(height, width):
#     shape = (height, width)

#     triangle_location = get_random_location(*shape)
#     circle_location1 = get_random_location(*shape, zoom=0.7)
#     circle_location2 = get_random_location(*shape, zoom=0.5)
#     mesh_location = get_random_location(*shape)
#     square_location = get_random_location(*shape, zoom=0.8)
#     plus_location = get_random_location(*shape, zoom=1.2)

#     # Create input image
#     arr = np.zeros(shape, dtype=bool)
#     arr = add_triangle(arr, *triangle_location)
#     arr = add_circle(arr, *circle_location1)
#     arr = add_circle(arr, *circle_location2, fill=True)
#     arr = add_mesh_square(arr, *mesh_location)
#     arr = add_filled_square(arr, *square_location)
#     arr = add_plus(arr, *plus_location)
#     arr = np.reshape(arr, (1, height, width)).astype(np.float32)

#     # Create target masks
#     masks = np.asarray([
#         add_filled_square(np.zeros(shape, dtype=bool), *square_location),
#         add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
#         add_triangle(np.zeros(shape, dtype=bool), *triangle_location),
#         add_circle(np.zeros(shape, dtype=bool), *circle_location1),
#          add_filled_square(np.zeros(shape, dtype=bool), *mesh_location),
#         # add_mesh_square(np.zeros(shape, dtype=bool), *mesh_location),
#         add_plus(np.zeros(shape, dtype=bool), *plus_location)
#     ]).astype(np.float32)

#     return arr, masks


# def add_square(arr, x, y, size):
#     s = int(size / 2)
#     arr[x-s,y-s:y+s] = True
#     arr[x+s,y-s:y+s] = True
#     arr[x-s:x+s,y-s] = True
#     arr[x-s:x+s,y+s] = True

#     return arr


# def add_filled_square(arr, x, y, size):
#     s = int(size / 2)

#     xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

#     return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))


# def logical_and(arrays):
#     new_array = np.ones(arrays[0].shape, dtype=bool)
#     for a in arrays:
#         new_array = np.logical_and(new_array, a)

#     return new_array


# def add_mesh_square(arr, x, y, size):
#     s = int(size / 2)

#     xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

#     return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))


# def add_triangle(arr, x, y, size):
#     s = int(size / 2)

#     triangle = np.tril(np.ones((size, size), dtype=bool))

#     arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

#     return arr


# def add_circle(arr, x, y, size, fill=False):
#     xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
#     circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
#     new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

#     return new_arr


# def add_plus(arr, x, y, size):
#     s = int(size / 2)
#     arr[x-1:x+1,y-s:y+s] = True
#     arr[x-s:x+s,y-1:y+1] = True

#     return arr


# def get_random_location(width, height, zoom=1.0):
#     x = int(width * random.uniform(0.1, 0.9))
#     y = int(height * random.uniform(0.1, 0.9))

#     size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

#     return (x, y, size)



def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        imgArray = img_array[i].transpose(1,2,0) # Reshape to match (H,W,C) from (C,H,W)
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(imgArray, cmap='gray')

    plt.savefig('pictures.png')
    
    


def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))



def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


def get_data_loaders():
    # use the same transformations for train/val in this example
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    # ])

    # train_set = SimDataset(100, transform = trans)
    # val_set = SimDataset(20, transform = trans)

    # image_datasets = {
    #     'train': train_set, 'val': val_set
    # }

    # batch_size = 25

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
    # pred = F.softmax(pred ,dim=3)
    # print(pred.shape)

    # # for x in range(192):
    # #      for y in range(192):
    # #           value1 =pred[]

    # # Find the index of the channel with the highest probability for each pixel
    # class_indices = torch.argmax(pred, dim=1)

    # print(class_indices.shape)
    # print(class_indices)
    # # Create a new tensor with a single channel (grayscale)
    # new_image = torch.zeros((1, 192, 192), dtype=torch.float32)

    # # Define grayscale values for each class
    # gray_values = {
    #     0: 0.0,     # Black for class 0
    #     1: 0.5020,  # Grey for class 1
    #     2: 1.0      # White for class 2
    # }

    # # Assign grayscale values based on the class with the highest probability
    # for class_index, gray_value in gray_values.items():
    #     new_image[class_indices == class_index] = gray_value
    
    # print(new_image.shape)
    # print(pred.shape)
    # print(target.shape)

    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
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
                scheduler.step()
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
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=5)

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
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape)

    # Change channel-order and make 3 channels for matplot
    # input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    # Map each channel (i.e. class) to each color
    # target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
    # pred_rgb = [masks_to_colorimg(x) for x in pred]

    plot_side_by_side([inputs.cpu().numpy(),  labels.cpu().numpy(), pred])