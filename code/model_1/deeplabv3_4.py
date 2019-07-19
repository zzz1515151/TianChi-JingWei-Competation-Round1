import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from .resnet import ResNet50_OS16
from .ASPP import ASPP_1_Bottleneck

class DeepLabV3_4(nn.Module):
    def __init__(self, opt):
        super(DeepLabV3_4, self).__init__()

        self.num_classes = opt['num_classes']


        self.resnet = ResNet50_OS16() # NOTE! specify the type of ResNet here
        self.aspp = ASPP_1_Bottleneck() # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        self.top_conv = nn.Conv2d(256, self.num_classes, kernel_size=1)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        feature_map = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))
        output = self.top_conv(feature_map)

        output = F.interpolate(output, size = [h, w], mode = 'bilinear', align_corners = False) # (shape: (batch_size, num_classes, h, w))

        return output

