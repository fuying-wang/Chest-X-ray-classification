FPNModels，NasnetModel, ResnetModels, multipath_resnet, Inceptionv2, DensenetModels等均为不同的网络结构，
按照同样的方式在Main.py和ChexnetTrainer.py中进行修改即可，区别是使用multipath_resnet时输入图片大小为1024*1024，即不进行crop和resize;
net_sphere.py为新的loss函数；实验中未使用；
HeatmapGenerator.py为产生热图的工具；
qt.py为GUI代码，linux下运行python qt.py即可使用；
实验环境为python2.7, pytorch 0.3.0, cuda8.0, ubuntu 16.04
