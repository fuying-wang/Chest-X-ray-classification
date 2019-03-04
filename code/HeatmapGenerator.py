import os
import numpy as np
import time
import sys
from PIL import Image

import cv2
from skimage impoer color

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

features_blobs = []


#---- Class to generate heatmaps (CAM)
class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):
       
        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True).cuda()
        elif nnArchitecture == 'ResNet-50': model = ResNet50(nnClassCount, True).cuda()
        elif nnArchitecture == 'ResNet-18': model = ResNet18(nnClassCount, True).cuda()
  
        model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])
        self.model = model.module.resnet18
        self.model.eval()
        

        #---- Initialize the weights
        params = list(self.model.parameters())
        self.weights = np.squeeze(params[-2].data.cpu().numpy())
        #self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
      
    #------------------------------------------------------------------------------
    def hook_feature(self, module, input, output):
        features_blobs.append(output)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        #---- Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
        self.model._modules.get('layer4').register_forward_hook(self.hook_feature)
        output = self.model(input.cuda())
        h_x = output.data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        #---- Generate heatmap
        heatmap = []
        feature_conv = features_blobs[0].data.cpu().numpy()
        bz, nc, h, w = feature_conv.shape
        size_upsample = (256, 256)
        #for i in range (0, len(self.weights)):
        #    map = features_blobs[0][0,i,:,:]
        #    if i == 0: heatmap = self.weights[i] * map
        #    else: heatmap += self.weights[i] * map
        for i in [idx[0]]: 
            cam = self.weights[i].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            heatmap.append(cam_img)
            #heatmap.append(cv2.resize(cam_img, size_upsample))
        
        #---- Blend original and heatmap 
        #npHeatmap = heatmap.cpu().data.numpy()
        npHeatmap = heatmap

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop)
        CAM = cv2.applyColorMap(cv2.resize(heatmap[0],(transCrop, transCrop)), cv2.COLORMAP_JET)
        
        mask = cv2.resize(heatmap[0],(transCrop, transCrop))
        #mask = mask / np.max(mask)
        mask = color.rgb2gray(mask)
        t = 0.7
        mask = np.maximum(mask, t)
        img = CAM * 0.3 + imgOriginal * 0.5
        cv2.imwrite(pathOutputFile, img)
#-------------------------------------------------------------------------------- 

pathInputImage = 'test/00009285_000.png'
pathOutputImage = 'test/heatmap.png'
pathModel = 'ResNet-18-m-21052018-160110.pth.tar'

nnArchitecture = 'ResNet-18'
nnClassCount = 14

transCrop = 224

h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
h.generate(pathInputImage, pathOutputImage, transCrop)
