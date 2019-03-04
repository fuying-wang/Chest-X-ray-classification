import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models
import torchvision.transforms as transforms

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from ResnetModels import *

#-------------------------------------------------------------------------------- 
pathInputImage = 'test/00009285_000.png'
pathOutputImage = 'test/heatmap.png'
pathModel = 'ResNet-50-m-17052018-210358.pth.tar'

nnArchitecture = 'ResNet-50'
nnClassCount = 14

transCrop = 224

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output)

if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True).cuda()
elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True).cuda()
elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True).cuda()
elif nnArchitecture == 'ResNet-50': model = ResNet50(nnClassCount, True).cuda()

model = torch.nn.DataParallel(model).cuda()

modelCheckpoint = torch.load(pathModel)
model.load_state_dict(modelCheckpoint['state_dict'])
model = model.module.resnet50
model.eval()

#---- Initialize the image transform - resize + normalize
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.Resize(transCrop))
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
        
transformSequence = transforms.Compose(transformList)
      
#---- Load image, transform, convert 
imageData = Image.open(pathInputImage).convert('RGB')
imageData = transformSequence(imageData)
imageData = imageData.unsqueeze_(0)
        
input = torch.autograd.Variable(imageData)
model._modules.get('layer4').register_forward_hook(hook_feature)
output = model(input.cuda())
h_x = output.data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()
        
feature_conv = features_blobs[0].data.cpu().numpy()
bz, nc, h, w = feature_conv.shape
H = np.max(np.abs(feature_conv), axis = 1)
print(H)
