import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import pretrainedmodels

class Nasnet(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(Nasnet, self).__init__()
		
        self.nasnet = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')

        kernelCount = self.nasnet.last_linear.in_features
		
        self.nasnet.last_linear = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.nasnet(x)
        return x

