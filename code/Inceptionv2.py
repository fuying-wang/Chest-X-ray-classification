import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import pretrainedmodels

class inceptionv2(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(inceptionv2, self).__init__()
		
        self.inceptionv2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')

        kernelCount = self.inceptionv2.last_linear.in_features
		
        self.inceptionv2.last_linear = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.inceptionv2(x)
        return x

