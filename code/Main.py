import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer

#-------------------------------------------------------------------------------- 

def main ():
    
    runTest()
    #runTrain()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = '../dataset/ChextXRay'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    
    pathFileTrain = './dataset/train_val_label.txt'
    pathFileVal = './dataset/val_label.txt'
    pathFileTest = './dataset/test_label.txt'
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = 'ResNet-18'
    #nnArchitecture = 'nasnetalarge'
    nnIsTrained = True
    nnClassCount = 14
    checkpoint =  'ResNet-18-m-21052018-160110.pth.tar'
    #checkpoint = 'm-17052018-143018.pth.tar'
    #checkpoint = './m-17052018-144133.pth.tar'
    #---- Training settings: batch size, maximum number of epochs

    trBatchSize = 32
    trMaxEpoch = 100
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
    
    #pathModel = 'm-' + timestampLaunch + '.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    Trainer = ChexnetTrainer()
    Trainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, \
                  trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, checkpoint)
    
    print ('Testing the trained model')
    Trainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = '../dataset/ChextXRay'
    pathFileTest = './dataset/test_label.txt'
    #nnArchitecture = 'DENSE-NET-121'
    nnArchitecture = 'ResNet-18'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 32
    imgtransResize = 256
    imgtransCrop = 224
    
    #pathModel = 'ResNet-50-m-17052018-184342.pth.tar'
    pathModel = 'ResNet-18-m-21052018-224624.pth.tar'
    timestampLaunch = ''
    Trainer = ChexnetTrainer()
    Trainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





