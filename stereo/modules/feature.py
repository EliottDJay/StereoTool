import copy

import torch
import torch.nn as nn
import timm
from stereo.modules.submodule import Conv2x, BasicConv, SubModule, SameConv2d
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


######################################################################
# ********************* utilization in HITNet  **********************
# HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching
# https://github.com/zjjMaiMai/TinyHITNet
# ####################################################################

class UpsampleBlock(nn.Module):
    # page 4
    # One up-sampling block applies 2 × 2 stride 2 transpose convolutions
    # to up-sample results of coarser U-Net resolution.
    # Features are concatenated with skip-connection, and a 1 × 1 convolution followed by
    # a 3 × 3 convolution are applied to merge the skipped and upsampled feature for the current resolution.
    def __init__(self, c0, c1):
        super(UpsampleBlock, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, 2, 2),
            nn.LeakyReLU(0.2),
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(c1, c1, 3, 1, 1),
            # nn.LeakyReLU(0.2),
        )

    def forward(self, input, sc):
        x = self.up_conv(input)
        if x.size()[2:] != sc.size()[2:]:
            x = x[:, :, : sc.size(2), : sc.size(3)]
        x = torch.cat((x, sc), dim=1)
        x = self.merge_conv(x)
        return x


class HITNET_Feature(SubModule):
    def __init__(self, cfg):
        super(HITNET_Feature, self).__init__()
        # self.type = cfg['type']
        self.chans = cfg['channels']
        # HITNet page3:
        # The network is composed of strided convolutions and transposed convolutions with leaky ReLUs as
        # non-linearities.
        # HITNet page 4 : one down-sampling block of the U-Net has a single 3 × 3 convolution followed by a
        # 2 × 2 convolution with stride 2.
        self.down_x1 = nn.Sequential(nn.Conv2d(3, self.chans[0], 3, 1, 1),
                                     nn.LeakyReLU(0.2),)

        self.down_x2 = nn.Sequential(
            SameConv2d(self.chans[0], self.chans[1], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.chans[1], self.chans[1], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_x4 = nn.Sequential(
            SameConv2d(self.chans[1], self.chans[2], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.chans[2], self.chans[2], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_x8 = nn.Sequential(
            SameConv2d(self.chans[2], self.chans[3], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.chans[3], self.chans[3], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_x16 = nn.Sequential(
            SameConv2d(self.chans[3], self.chans[4], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.chans[4], self.chans[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(chans[4], chans[4], 3, 1, 1),
            # nn.LeakyReLU(0.2),
            # nn.Conv2d(chans[4], chans[4], 3, 1, 1),
            # nn.LeakyReLU(0.2),
        )
        # weather to use a deeper feature net like the github project
        self.up_x8 = UpsampleBlock(self.chans[4], self.chans[3])
        self.up_x4 = UpsampleBlock(self.chans[3], self.chans[2])
        self.up_x2 = UpsampleBlock(self.chans[2], self.chans[1])
        self.up_x1 = UpsampleBlock(self.chans[1], self.chans[0])

    def ccount(self):
        # 1 1/2 1/4 1/8 1/6
        return self.chans

    def forward(self, input):
        x1 = self.down_x1(input)
        x2 = self.down_x2(x1)
        x4 = self.down_x4(x2)
        x8 = self.down_x8(x4)
        out_16 = self.down_x16(x8)
        out_8 = self.up_x8(out_16, x8)
        out_4 = self.up_x4(out_8, x4)
        out_2 = self.up_x2(out_4, x2)
        out_1 = self.up_x1(out_2, x1)

        return [out_16, out_8, out_4, out_2, out_1]



