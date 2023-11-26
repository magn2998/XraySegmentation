# Source: https://github.com/Mostafa-wael/U-Net-in-PyTorch/blob/main/U_Net.ipynb

# First, the necessary modules are imported from the torch and torchvision packages, including the nn module for building neural networks and the pre-trained models provided in torchvision.models. 
# The relu function is also imported from torch.nn.functional.
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu, softmax

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm = False):
        super().__init__()
        if(use_batchnorm):
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
             self.conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size = 3, padding = 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, inputs):
        x = self.conv(inputs)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

    def forward(self, inputs):
        x = self.conv(inputs) #Output of conv block used for skip
        p = self.pool(x) #output of pooling
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels+out_channels, out_channels)
        

    def forward(self, inputs, skip):
        x = self.convTrans(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


# Then, a custom class UNet is defined as a subclass of nn.Module. 
# The __init__ method initializes the architecture of the U-Net by defining the layers for both the encoder and decoder parts of the network. 
# The argument n_class specifies the number of classes for the segmentation task.
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        #Filters for the layers
        filters = [64, 128, 256, 512, 1024]
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e1 = EncoderBlock(1,filters[0])

        # input: 284x284x64
        self.e2 = EncoderBlock(filters[0],filters[1])

        # input: 140x140x128
        self.e3 = EncoderBlock(filters[1],filters[2])

        # input: 68x68x256
        self.e4 = EncoderBlock(filters[2],filters[3])

        #Bottleneck
        # input: 32x32x512
        self.b = DoubleConv(filters[3],filters[4])
        
        # Decoder
        # In the decoder, transpose convolutional layers with the ConvTranspose2d function are used to upsample the feature maps to the original size of the input image. 
        # Each block in the decoder consists of an upsampling layer, a concatenation with the corresponding encoder feature map, and two convolutional layers.
        # -------

        self.d1 = DecoderBlock(filters[4], filters[3])

        self.d2 = DecoderBlock(filters[3], filters[2])

        self.d3 = DecoderBlock(filters[2], filters[1])
        
        self.d4 = DecoderBlock(filters[1], filters[0])

        # Output layer
        self.outconv = nn.Conv2d(filters[0], n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck
        b = self.b(p4)
        
        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Output layer
        out = self.outconv(d4)

        return out

import test

test.run(UNet)



