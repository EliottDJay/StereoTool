import copy

import torch
import torch.nn as nn
import timm
from stereo.modules.submodule import Conv2x, BasicConv, SubModule
"""
channels:
  mobilenetv3_large_100: [16,24,40,112,160]
  mobilenetv2_120d: [24,32,40,112,192]
  mobilenetv2_100: [16,24,32,96,160]
  mnasnet_100: [16,24,40,96,192]
  efficientnet_b0: [16,24,40,112,192]
  efficientnet_b3a: [24,32,48,136,232]
  mixnet_xl: [40,48,64,192,320]
  dla34: [32,64,128,256,512]

layers:
  mobilenetv3_large_100: [1,2,3,5,6]
  mobilenetv2_120d: [1,2,3,5,6]
  mobilenetv2_100: [1,2,3,5,6]
  mnasnet_100: [1,2,3,5,6]
  efficientnet_b0: [1,2,3,5,6]
  efficientnet_b3a: [1,2,3,5,6]
  mixnet_xl: [1,2,3,5,6]
  dla34: [1,2,3,5,6]
"""


# mobilev2, mobilev3
class BasicFeature(nn.Module):
    def __init__(self, cfg):
        super(BasicFeature, self).__init__()
        # self.cfg = cfg
        self.type = cfg['type']
        chans = cfg['channels']
        layers = cfg['layers']

        pretrained = cfg['pretrained']
        model = timm.create_model(self.type, pretrained=pretrained, features_only=True)

        # https://medium.com/red-buffer/getting-started-with-pytorch-distributed-54ae933bb9f0
        # timm 只需要最后上 DDP就行了

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        x2 = self.block0(x)
        x4 = self.block1(x2)

        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x_out = [x4, x8, x16, x32]

        return x2, x_out


class BasicFeaUp(SubModule):
    """
    keep channel of img 1,2,3 double
    """
    def __init__(self, cfg):
        super(BasicFeaUp, self).__init__()
        # self.cfg = cfg
        self.type = cfg['type']
        self.chans = cfg['channels']

        self.deconv32_16 = Conv2x(self.chans[4], self.chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(self.chans[3] * 2, self.chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(self.chans[2] * 2, self.chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(self.chans[1] * 2, self.chans[1] * 2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def ccount(self):
        channls = copy.deepcopy(self.chans)
        for i in [3, 2, 1]:
            channls[i] = channls[i]*2
        # 1/2 1/4 1/8 1/16 1/32
        # print(self.chans, channls, '--------------------------feature')
        return self.chans, channls

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        if featR is not None:
            y4, y8, y16, y32 = featR

            x16 = self.deconv32_16(x32, x16)
            y16 = self.deconv32_16(y32, y16)

            x8 = self.deconv16_8(x16, x8)
            y8 = self.deconv16_8(y16, y8)

            x4 = self.deconv8_4(x8, x4)
            y4 = self.deconv8_4(y8, y4)

            x4 = self.conv4(x4)
            y4 = self.conv4(y4)

            return [x4, x8, x16, x32], [y4, y8, y16, y32]
        else:
            x16 = self.deconv32_16(x32, x16)
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x4 = self.conv4(x4)

            # chans[1]*2, chans[1] chans[2] chans[3]
            return [x4, x8, x16, x32]


