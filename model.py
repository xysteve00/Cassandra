import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class TrafficSignModel(nn.Module):
    """Feedforward traffice sign neural network"""
    def __init__(self, cct=False, base=32, dense=512, num_classes=43):
        super().__init__()
        # Conv layers
        if cct:                
            self.stem = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(15,base,3, padding=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(base,base,3, padding=1)),
            ('relu2', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(p=0.2)),
            ('conv3', nn.Conv2d(base,2*base,3, padding=1)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(2*base,2*base,3, padding=1)),
            ('relu4', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(2)),
            ('dropout2', nn.Dropout2d(p=0.2)),
            ('conv5', nn.Conv2d(2*base,4*base,3, padding=1)),
            ('relu5', nn.ReLU()),
            ('conv6', nn.Conv2d(4*base,4*base,3, padding=1)),
            ('relu6', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(2)),
            ('dropout3', nn.Dropout2d(p=0.2)),
            ('avgpool', nn.AdaptiveAvgPool2d((4,4))),
            ('flatten', nn.Flatten()),        ]))   
        else:             
            self.stem = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3,base,3, padding=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(base,base,3, padding=1)),
            ('relu2', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(p=0.2)),
            ('conv3', nn.Conv2d(base,2*base,3, padding=1)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(2*base,2*base,3, padding=1)),
            ('relu4', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(2)),
            ('dropout2', nn.Dropout2d(p=0.2)),
            ('conv5', nn.Conv2d(2*base,4*base,3, padding=1)),
            ('relu5', nn.ReLU()),
            ('conv6', nn.Conv2d(4*base,4*base,3, padding=1)),
            ('relu6', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(2)),
            ('dropout3', nn.Dropout2d(p=0.2)),
            ('avgpool', nn.AdaptiveAvgPool2d((4,4))),
            ('flatten', nn.Flatten()),        ]))   

        self.fc_dropout = nn.Dropout(p=0.0)
        self.projection = nn.Linear((4*4)*(4*base), dense) # output layer
        #self.classifier = nn.Linear(dense,num_classes)    

    def forward(self, x):

        # F.max_pool2d()
        x = self.stem(x)
        x = F.relu(self.projection(x))
        x = self.fc_dropout(x)
        x = self.classifier(x)       
        return x

class MnistModel(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # conv layer
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        # hidden layer
        self.linear1 = nn.Linear(320, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)

        
    def forward(self, xb):
        #input size
        input_size = xb.size(0)
        xb = F.relu(self.mp(self.conv1(xb)))
        xb = F.relu(self.mp(self.conv2(xb)))
        # Flatten the image tensors
        xb = xb.view(input_size, -1)
        # Get intermediate outputs using hidden layer
        out1 = self.linear1(xb)
        # output embedding
        _out = self.linear2(out1)
        # Apply activation function
        out = F.log_softmax(self.linear2(out1), dim=1)
        return out


class ModdedLeNet5Net(nn.Module):
    """
    A modified LeNet architecture that seems to be easier to embed backdoors in than the network from the original
    badnets paper
    Input - (1 or 3)x28x28
    C1 - 6@28x28 (5x5 kernel)
    ReLU
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    ReLU
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    ReLU
    F7 - 10 (Output)
    """

    def __init__(self, channels=3):
        super(ModdedLeNet5Net, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 6, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class BadNetExample(nn.Module):
    """
    Mnist network from BadNets paper
    Input - 1x28x28
    C1 - 1x28x28 (5x5 kernel) -> 16x24x24
    ReLU
    S2 - 16x24x24 (2x2 kernel, stride 2) Subsampling -> 16x12x12
    C3 - 16x12x12 (5x5 kernel) -> 32x8x8
    ReLU
    S4 - 32x8x8 (2x2 kernel, stride 2) Subsampling -> 32x4x4
    F6 - 512 -> 512
    tanh
    F7 - 512 -> 10 Softmax (Output)
    """

    def __init__(self):
        super(BadNetExample, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
