import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from model import *
from collections import OrderedDict
import geffnet

def load_pretrained_models(model_type,cct=False):
    if model_type == 'ts':
        if cct:
            net = TrafficSignModel(cct=True, num_classes=2)  # Build a CNN model
        else:
            net = TrafficSignModel(cct=False, num_classes=2)  # Build a CNN model
    # resnet
    elif model_type == 'resnet':
        net = models.resnet18(pretrained=False)
        net.fc = nn.Linear(512, 2)
    elif model_type == 'LeNet':
        net = ModdedLeNet5Net()
        net.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2)
        )
    elif model_type == 'efficientnetb3':
        net = geffnet.create_model('efficientnet_b3', pretrained=False)
    elif model_type == 'mobilenetv3':
        # net = geffnet.create_model('efficientnet_b3', pretrained=True)
        if cct:
            net = geffnet.create_model('mobilenetv3_rw', pretrained=True)
            net.conv_stem = nn.Conv2d(15, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        else:
            net = geffnet.create_model('mobilenetv3_rw', pretrained=True)
            net.classifier = nn.Linear(in_features=1280, out_features=2, bias=True)
    # VGG
    elif model_type == 'vgg':
        net = models.vgg16(pretrained=False)
        net.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(4096, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(2048, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 43))
    # Alex
    elif model_type == 'alex':
        net = models.alexnet(pretrained=False)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=43, bias=True))
    # densenet
    elif model_type == 'densenet':
        net = models.densenet161(pretrained=False)
        net.classifier = nn.Sequential(nn.Linear(2208, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 2))
    elif model_type == 'googlenet':
        net = models.googlenet(pretrained=False)
        net.fc = nn.Sequential(nn.Linear(1024, 2))

    else:
        print('ERROR: model type NOT FOUND')
        return
    
    return net
