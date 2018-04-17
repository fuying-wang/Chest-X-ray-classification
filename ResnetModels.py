import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(ResNet18, self).__init__()
		
        self.resnet18 = torchvision.models.resnet18(pretrained=False)

        kernelCount = self.resnet18.fc.in_features
		
        self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet18(x)
        return x


class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(ResNet50, self).__init__()
		
        self.resnet50 = torchvision.models.resnet50(pretrained=True)

        kernelCount = self.resnet50.fc.in_features
		
        self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
