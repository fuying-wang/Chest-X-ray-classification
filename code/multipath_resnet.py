import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class mp_resnet(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(mp_resnet, self).__init__()
        
        self.resnet101 = torchvision.models.resnet101(pretrained=False)

        self.path1 = nn.Sequential(self.resnet101.layer1, self.resnet101.layer2, self.resnet101.layer3, self.resnet101.layer4)
        self.path2 = nn.Sequential(self.resnet101.layer1, self.resnet101.layer2, self.resnet101.layer3)
        self.path3 = nn.Sequential(self.resnet101.layer1, self.resnet101.layer2, self.resnet101.layer3)

        self.po1 = nn.AdaptiveAvgPool2d(224)
        self.po2 = nn.AdaptiveAvgPool2d(512)
        self.po3 = nn.AdaptiveAvgPool2d(1024)
        self.po4 = nn.AdaptiveAvgPool2d(32)
        self.po5 = nn.AdaptiveAvgPool2d(32)
        self.po6 = nn.AdaptiveAvgPool2d(32)
        self.fc = nn.Sequential(nn.Linear(1024,18),nn.Sigmoid())


    def forward(self, x):
        y3 = nn.Sequential(self.po3, self.path3, self.po4)
        y2 = nn.Sequential(self.po2, self.path2, self.po5)
        y1 = nn.Sequential(self.po1, self.path1, self.po6)
        y = nn.concat(y1,y2,y3,dims=-1)
        y = self.fc(y)

        return y


