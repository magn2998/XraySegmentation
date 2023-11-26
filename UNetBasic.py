# This is a test implementation of the UNet architechture


# Steps
# 0. Import all the good stuff
# 1. Load data
# 2. Augment data -> crop to be 128x128
# 3. Setup loss and optimizer
# 4. Define model -> forward and backward function
# 5. Train the model
# 6. Test it
# Seems easy huh? U wrong


# 0
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import rich
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from sklearn import metrics

# Determine which device to use
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
rich.print(f"Device: [red]{DEVICE}")


# Load the data




# Helper functions for evaluation
def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())


def compute_confusion_matrix(target, pred, normalize=None):
    return metrics.confusion_matrix(
        target.detach().cpu().numpy(),
        pred.detach().cpu().numpy(),
        normalize=normalize
    )


def show_image(img):
    img = img.detach().cpu()
    img = img / 2 + 0.5  # unnormalize
    with sns.axes_style("white"):
        plt.figure(figsize=(8, 8))
        plt.imshow(img.permute((1, 2, 0)).numpy())
        plt.axis('off')
        plt.show()


batch_size = 64  # both for training and testing


# The UNet comprises an encoder block a bottleneck and a decoder block
# The special thing here is that it contains skip connections between the encoder and decoder

# The encoder block
def encoder(inputs, num_filters):
    x = nn.Conv2d(inputs.shape[1], num_filters, kernel_size=3, padding=0)(inputs)
    x = nn.ReLU()(x)
    x = nn.Conv2d(len(x), num_filters, kernel_size=3, padding=0)(x)
    x = nn.ReLU()(x)
    x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
    return x

# The decoder block
def decoder(inputs, skip_features, num_filters):
    x = nn.ConvTranspose2d(inputs.shape[1], num_filters, kernel_size=2, stride=2, padding=0)(inputs)

