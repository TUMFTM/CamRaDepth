import os
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3
from torchinfo import summary
from models.model_utils import Unpool, weights_init, weights_init_kaiming, weights_init_kaiming_leaky
from models.model_utils import BasicBlock, DSConv_Block, Decoder, DeConv, UpConv, UpProj, choose_decoder
import params
import collections
import numpy as np
import matplotlib.pyplot as plt

class Stage1(nn.Module):
    def __init__(self, decoder, output_size, in_channels):
        super(Stage1, self).__init__()
        self.output_size = output_size
        self.inplanes = 32
        self.in_channels = in_channels
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1
        self.base_width = 16

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        weights_init_kaiming_leaky(self.conv1)
        weights_init_kaiming(self.bn1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, kernel_size=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, kernel_size=3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, kernel_size=3, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, kernel_size=3, stride=2)
    
        num_channels = 512

    ## Seg Decoder ###
        self.decoder_seg = choose_decoder(decoder, num_channels)

        self.decoder_seg.layer1 = self.decoder_seg.UpProjModule(num_channels, num_channels//2)
        if params.skip_con[2]:
            inplanes = num_channels//2 + num_channels//2
            self.decoder_seg.layer1_conv = BasicBlock(inplanes, num_channels//2, kernel_size=3, stride=1)
        else:
            self.decoder_seg.layer1_conv = nn.Sequential(collections.OrderedDict([
                    ('Conv',nn.Conv2d(num_channels//2, num_channels//2, kernel_size=3, stride=1, padding=1)),
                    ('Norm',self._norm_layer(num_channels//2)),
                    ('Relu',nn.LeakyReLU(0.2, inplace=True))
                ]))
        self.decoder_seg.layer2 = self.decoder_seg.UpProjModule(num_channels//2, num_channels//4)
        if params.skip_con[1]:
            inplanes = num_channels//4 + num_channels//4
            self.decoder_seg.layer2_conv = BasicBlock(inplanes, num_channels//4, kernel_size=3, stride=1, res=False)
        else:
            self.decoder_seg.layer2_conv = nn.Sequential(collections.OrderedDict([
                    ('Conv',nn.Conv2d(num_channels//4, num_channels//4, kernel_size=3, stride=1, padding=1)),
                    ('Norm',self._norm_layer(num_channels//4)),
                    ('Relu',nn.LeakyReLU(0.2, inplace=True))
                ]))
       
        self.decoder_seg.layer3 = self.decoder_seg.UpProjModule(num_channels//4, num_channels//8)
        if params.skip_con[0]:
            inplanes = num_channels//8 + num_channels//8
            self.decoder_seg.layer3_conv = nn.Sequential(collections.OrderedDict([
                    ('BasicBlock', BasicBlock(inplanes, num_channels//8, kernel_size=3, stride=1, res=False)),
                    ('Conv',nn.Conv2d(num_channels//8, num_channels//16, kernel_size=3, stride=1, padding=1)),
                    ('Norm',self._norm_layer(num_channels//16)),
                    ('Relu',nn.LeakyReLU(0.2, inplace=True))
                ]))
        else:
            self.decoder_seg.layer3_conv = nn.Sequential(collections.OrderedDict([
                    ('Conv',nn.Conv2d(num_channels//8, num_channels//16, kernel_size=3, stride=1, padding=1)),
                    ('Norm',self._norm_layer(num_channels//16)),
                    ('Relu',nn.LeakyReLU(0.2, inplace=True))
                ]))
                
        
        self.conv3_seg= nn.Conv2d(num_channels//16,params.n_classes,kernel_size=3,stride=1,padding=1,bias=False)
        # weight init
        self.decoder_seg.apply(weights_init)
        self.conv3_seg.apply(weights_init)


    ### Depth Decoder ###
        self.decoder_depth = choose_decoder(decoder, num_channels)

        self.decoder_depth.layer1 = self.decoder_depth.UpProjModule(num_channels, num_channels//2)
        if params.skip_con[2]:
            inplanes = num_channels//2 + num_channels//2
            self.decoder_depth.layer1_conv = BasicBlock(inplanes, num_channels//2, kernel_size=3, stride=1)
        else:
            self.decoder_depth.layer1_conv = nn.Sequential(collections.OrderedDict([
                    ('Conv',nn.Conv2d(num_channels//2, num_channels//2, kernel_size=3, stride=1, padding=1)),
                    ('Norm',self._norm_layer(num_channels//2)),
                    ('Relu',nn.LeakyReLU(0.2, inplace=True))
                ]))
        self.decoder_depth.layer2 = self.decoder_depth.UpProjModule(num_channels//2, num_channels//4)
        if params.skip_con[1]:
            inplanes = num_channels//4 + num_channels//4
            self.decoder_depth.layer2_conv = BasicBlock(inplanes, num_channels//4, kernel_size=3, stride=1, res=False)
        else:
            self.decoder_depth.layer2_conv = nn.Sequential(collections.OrderedDict([
                    ('Conv',nn.Conv2d(num_channels//4, num_channels//4, kernel_size=3, stride=1, padding=1)),
                    ('Norm',self._norm_layer(num_channels//4)),
                    ('Relu',nn.LeakyReLU(0.2, inplace=True))
                ]))

        self.decoder_depth.layer3 = self.decoder_depth.UpProjModule(num_channels//4, num_channels//8)
        if params.skip_con[0]:
            inplanes = num_channels//8 + num_channels//8
            self.decoder_depth.layer3_conv = nn.Sequential(collections.OrderedDict([
                    ('BasicBlock', BasicBlock(inplanes, num_channels//8, kernel_size=3, stride=1, res=False)),
                    ('Conv',nn.Conv2d(num_channels//8, num_channels//16, kernel_size=3, stride=1, padding=1)),
                    ('Norm',self._norm_layer(num_channels//16)),
                    ('Relu',nn.LeakyReLU(0.2, inplace=True))
                ]))
        else:
            self.decoder_depth.layer3_conv = nn.Sequential(collections.OrderedDict([
                    ('Conv',nn.Conv2d(num_channels//8, num_channels//16, kernel_size=3, stride=1, padding=1)),
                    ('Norm',self._norm_layer(num_channels//16)),
                    ('Relu',nn.LeakyReLU(0.2, inplace=True))
                ]))
                
       
        self.conv3_depth= nn.Conv2d(num_channels//16,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        # weight init
        self.decoder_depth.apply(weights_init)
        self.conv3_depth.apply(weights_init)


    # Make layer function adapted from resnet
    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1, dilation=1, res=True):
        norm_layer = self._norm_layer
        downsample = None
        #if dilation > 1:
        #    dilation *= stride
        #    stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, stride, downsample,
                            dilation, norm_layer, res))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size,
                                dilation=dilation, norm_layer=norm_layer, res=res))

        layers = nn.Sequential(*layers)

        # Explicitly initialize layers after construction
        for m in layers.modules():
            weights_init_kaiming(m)

        return layers


    def forward(self, x):
       
        # ipdb.set_trace()

        x = self.conv1(x) # -> 16, 208, 400
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # -> 16, 104, 200
        skip_x_1 = x
        x = self.layer2(x)  # -> 32, 52, 100
        skip_x_2 = x
        x = self.layer3(x)  # -> 64, 26, 50
        skip_x_3 = x
        x = self.layer4(x)  # -> 128, 13, 25

        ### Seg Decoder ###
       
        y_seg = self.decoder_seg.layer1(x)
        if params.skip_con[2]:
            y_seg = torch.cat((y_seg, skip_x_3), dim=1)
        y_seg = self.decoder_seg.layer1_conv(y_seg)
        y_seg = self.decoder_seg.layer2(y_seg)
        if params.skip_con[1]:
            y_seg = torch.cat((y_seg, skip_x_2), dim=1)
        y_seg = self.decoder_seg.layer2_conv(y_seg)
        y_seg = self.decoder_seg.layer3(y_seg)
        if params.skip_con[0]:
            y_seg = torch.cat((y_seg, skip_x_1), dim=1)
        y_seg = self.decoder_seg.layer3_conv(y_seg)

        y_seg = self.conv3_seg(y_seg)

        ### Depth Decoder ###
        y_depth = self.decoder_depth.layer1(x)      
        if params.skip_con[2]:
            y_depth = torch.cat((y_depth, skip_x_3), dim=1)
        y_depth = self.decoder_depth.layer1_conv(y_depth)
        y_depth = self.decoder_depth.layer2(y_depth)
        if params.skip_con[1]:
            y_depth = torch.cat((y_depth, skip_x_2), dim=1)
        y_depth = self.decoder_depth.layer2_conv(y_depth)
        y_depth = self.decoder_depth.layer3(y_depth)      
        if params.skip_con[0]:
            y_depth = torch.cat((y_depth, skip_x_1), dim=1)
        y_depth = self.decoder_depth.layer3_conv(y_depth)

        y_depth = self.conv3_depth(y_depth)               
        y_depth = self.sigmoid(y_depth)             

        return [y_depth, y_seg]

class Stage2(nn.Module):
    def __init__(self, decoder, output_size, in_channels=2):
        super(Stage2, self).__init__()
        self.output_size = output_size
        self.inplanes = 1
        self.in_channels = in_channels
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1
        self.base_width = 16

        self.layer1_seg = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2_seg = self._make_layer(BasicBlock, 128, 2, stride=1)
        self.layer3_seg = self._make_layer(BasicBlock, 256, 2, stride=1)

        self.inplanes = 1
    
        self.layer1_depth = self._make_layer(BasicBlock, 64, 2, kernel_size=3, stride=1)
        self.layer2_depth = self._make_layer(BasicBlock, 128, 2, kernel_size=3, stride=1)
        self.layer3_depth = self._make_layer(BasicBlock, 256, 2, kernel_size=3, stride=1)

        n_encoders = 2        
        num_channels = 512

        self.conv_fusion =  nn.Sequential(collections.OrderedDict([
              ('conv1',     nn.Conv2d(256 * n_encoders, num_channels, kernel_size=1, bias=False)),
              ('batchnorm1', nn.BatchNorm2d(num_channels)),
              ('relu',       nn.LeakyReLU(0.2)), #nn.ReLU()),
              ('conv2',      BasicBlock(num_channels, num_channels, stride=1))
            ]))

        ### Depth Decoder ###
        self.decoder_depth = choose_decoder(decoder, num_channels)

        self.decoder_depth.layer1 = nn.Sequential(collections.OrderedDict([
                #('UpProj', self.decoder_depth.UpProjModule(num_channels, num_channels//2)),
                ('Conv1',conv3x3(num_channels, num_channels//2, stride=1)),
                ('Norm1',self._norm_layer(num_channels//2)),
                ('Relu1',nn.LeakyReLU(0.2, inplace=True)),
                ('Conv2',conv3x3(num_channels//2, num_channels//4, stride=1)),
                ('Norm2',self._norm_layer(num_channels//4)),
                ('Relu2',nn.LeakyReLU(0.2, inplace=True))
            ]))
        self.decoder_depth.layer2 = nn.Sequential(collections.OrderedDict([
                ('UpProj', self.decoder_depth.UpProjModule(num_channels//4, num_channels//8)),
                ('Conv',conv3x3(num_channels//8, num_channels//16, stride=1)),
                ('Norm',self._norm_layer(num_channels//16)),
                ('Relu',nn.LeakyReLU(0.2, inplace=True))
            ]))
      
        self.conv3 = nn.Conv2d(num_channels//16,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.bicubic = nn.Upsample(size=self.output_size, mode='bicubic', align_corners=True)

        # weight init
        self.decoder_depth.apply(weights_init)
        self.conv3.apply(weights_init)


    # Make layer function adapted from resnet
    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1, dilation=1, res=True):
        norm_layer = self._norm_layer
        downsample = None
       
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, stride, downsample, dilation, norm_layer, res))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size, dilation=dilation,
                                norm_layer=norm_layer, res=res))

        layers = nn.Sequential(*layers)

        # Explicitly initialize layers after construction
        for m in layers.modules():
            weights_init_kaiming(m)

        return layers


    def forward(self, x):
       
     
        x_depth = x[:,0,:,:]
        x_depth = x_depth.unsqueeze(1)
        x_seg = x[:,1,:,:]
        x_seg = x_seg.unsqueeze(1)


        x_seg = self.layer1_seg(x_seg)  # -> 16, 104, 200
        x_seg = self.layer2_seg(x_seg)  # -> 32, 52, 100
        x_seg = self.layer3_seg(x_seg)  # -> 64, 26, 50

        x_depth = self.layer1_depth(x_depth)  # -> 16, 104, 200
        skip_x_1 = x_depth 
        x_depth  = self.layer2_depth(x_depth)  # -> 32, 52, 100
        skip_x_2 = x_depth 
        x_depth  = self.layer3_depth(x_depth)  # -> 64, 26, 50
        skip_x_3 = x_depth 
        
        x = torch.cat((x_seg, x_depth), dim=1)
        x = self.conv_fusion(x)         # -> 512, 13, 25

        ### Decoder ###
     
        y = x
        y = self.decoder_depth.layer1(y)      # -> 128, 26, 50
        y = self.decoder_depth.layer2(y)      # -> 64, 52, 100

        y = self.conv3(y)               # -> 1, 208, 400
        y = self.sigmoid(y)
        #y = self.bilinear(y)            # -> 1, 416, 800

        return y

class Multitask_multistage(nn.Module):
    def __init__(self, decoder, output_size, in_channels):
        super(Multitask_multistage, self).__init__()

        self.in_channels = in_channels
        self.output_size = output_size

        self.in_channels_stage2 = 2 # 1 Seg + 1 Filtered Depth + 3 RGB
        
        self.stage1 = Stage1("upproj", self.output_size, self.in_channels)
        self.stage2 = Stage2("upproj", self.output_size, self.in_channels_stage2)

        self.transform = torchvision.transforms.Resize(size=(int(params.image_dimension[0]/2), int(params.image_dimension[1]/2)))

    def forward(self, x):

        if params.rgb_only:
            x = x[:, :3, :, :]
            
        y_stage1 = self.stage1(x)
        y_stage1_depth = y_stage1[0]
        y_stage1_seg = y_stage1[1]
        x_stage2_seg = torch.argmax(y_stage1_seg, dim=1)
        x_stage2_seg = x_stage2_seg.unsqueeze(1)
        
        x_stage2 = torch.cat((y_stage1_depth, x_stage2_seg), dim=1)

        y_stage2 = self.stage2(x_stage2)

        return [y_stage1_depth, y_stage1_seg, y_stage2]      

if __name__ == "__main__":

    # Create model
    feature_channels = 4
    if params.sparse_depth_uv:
        feature_channels += 2
    if params.im_uv:
        feature_channels += 2
    if params.rad_vel:
        feature_channels += 1

    if params.rgb_only:
        feature_channels = 3

    model = Multitask_multistage("upproj", output_size=[416, 800], in_channels=feature_channels).cuda()
    batch_size = 1
    summary(model,(batch_size, feature_channels, 416, 800))
    #print(model)