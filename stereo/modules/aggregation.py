import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from utils.args_dictionary import get_key, key_exist, get_present_caller
from utils.logger import Logger as Log

from stereo.modules.submodule import BasicConv, ChannelAtt, SubModule

class Hourglass(SubModule):
    def __init__(self, in_channels):
        super(Hourglass, self).__init__()
        self.conv1 = BasicConv(in_channels, in_channels * 2, deconv=False, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv(in_channels * 2, in_channels * 2, deconv=False, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)
        self.conv3 = BasicConv(in_channels * 2, in_channels * 4, deconv=False, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)
        self.conv4 = BasicConv(in_channels * 4, in_channels * 4, deconv=False, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)

        self.conv5 = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)
        self.conv6 = BasicConv(in_channels * 2, in_channels, deconv=True, is_3d=True, bn=True, relu=True,
                               kernel_size=3, stride=2, padding=1)

        self.redir1 = BasicConv(in_channels * 2, in_channels * 2, deconv=False, is_3d=True, bn=False, relu=False,
                               kernel_size=1, stride=1, padding=0)
        self.redir2 = BasicConv(in_channels, in_channels, deconv=False, is_3d=True, bn=False, relu=False,
                                kernel_size=1, stride=1, padding=0)

        self.acf = nn.LeakyReLU()
        # SubModule 的初始化
        self.weight_init()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = self.conv5(conv4) + self.redir1(conv2)
        conv5 = self.acf(conv5)
        conv6 = self.conv6(conv5) + self.redir2(x)
        conv6 = self.acf(conv6)
        return conv6


class GCE(SubModule):
    def __init__(self, cfg, max_disparity=192, matching_head=1, im_channels=None,
                 vchannels=[16, 32, 48], blocks_num=[2, 2, 2]):
        super(GCE, self).__init__()
        self.cfg = cfg
        if im_channels is None:
            Log.error('{} feature channel number should be calculated before initializing the GCE Aggregation block'
                      .format(get_present_caller()))
            exit(1)
        ichans = im_channels
        self.D = max_disparity  # GCE中是/4了的 但是实际上这里可以传入本身就弄好了的

        # stem for CoEx
        stemhead = 8
        self.conv_stem = BasicConv(matching_head, stemhead,
                                   is_3d=True, kernel_size=3, stride=1, padding=1)  # head 1 --> 8
        self.channelAttStem = ChannelAtt(stemhead, ichans[1], self.D)

        # downsampe x2 1/4 --> 1/8 head: 8-->16
        self.convdown1 = nn.Sequential(BasicConv(stemhead, vchannels[0], is_3d=True, bn=True, relu=True,
                                       kernel_size=3, padding=1, stride=(2, 2, 2), dilation=1),
                                       BasicConv(vchannels[0], vchannels[0], is_3d=True, bn=True, relu=True,
                                       kernel_size=3, padding=1, stride=1, dilation=1),
                                       )
        self.channelAttdown1 = ChannelAtt(vchannels[0], ichans[2], self.D//2)
        # downsampe x2 1/8 --> 1/16 head: 16-->32
        self.convdown2 = nn.Sequential(BasicConv(vchannels[0], vchannels[1], is_3d=True, bn=True, relu=True,
                                       kernel_size=3, padding=1, stride=(2, 2, 2), dilation=1),
                                       BasicConv(vchannels[1], vchannels[1], is_3d=True, bn=True, relu=True,
                                       kernel_size=3, padding=1, stride=1, dilation=1),
                                       )
        self.channelAttdown2 = ChannelAtt(vchannels[1], ichans[3], self.D // (2**2))
        # downsampe x2 1/16 --> 1/32 head: 32-->48
        self.convdown3 = nn.Sequential(BasicConv(vchannels[1], vchannels[2], is_3d=True, bn=True, relu=True,
                                                 kernel_size=3, padding=1, stride=(2, 2, 2), dilation=1),
                                       BasicConv(vchannels[2], vchannels[2], is_3d=True, bn=True, relu=True,
                                                 kernel_size=3, padding=1, stride=1, dilation=1),
                                       )
        self.channelAttdown3 = ChannelAtt(vchannels[2], ichans[4], self.D // (2 ** 3))

        # upsample x2  1/32 --> 1/16 head: 48-->32
        self.convup1 = BasicConv(vchannels[2], vchannels[1], deconv=True, is_3d=True, bn=True,
                                 relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.convskip1 = BasicConv(2*vchannels[1], vchannels[1], is_3d=True, kernel_size=1, padding=0, stride=1)
        self.conv_agg1 = nn.Sequential(
            BasicConv(
                vchannels[1], vchannels[1], is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(
                vchannels[1], vchannels[1], is_3d=True, kernel_size=3, padding=1, stride=1)
        )
        self.channelAttup1 = ChannelAtt(vchannels[1], ichans[3], self.D // (2 ** 2))
        # upsample x2  1/16 --> 1/8 head: 32-->16
        self.convup2 = BasicConv(vchannels[1], vchannels[0], deconv=True, is_3d=True, bn=True,
                                 relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.convskip2 = BasicConv(2 * vchannels[0], vchannels[0], is_3d=True, kernel_size=1, padding=0, stride=1)
        self.conv_agg2 = nn.Sequential(
            BasicConv(
                vchannels[0], vchannels[0], is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(
                vchannels[0], vchannels[0], is_3d=True, kernel_size=3, padding=1, stride=1)
        )
        self.channelAttup2 = ChannelAtt(vchannels[0], ichans[2], self.D //2)
        # upsample x2  1/8 --> 1/4 head: 16-->8
        self.convup3 = BasicConv(vchannels[0], 1, deconv=True, is_3d=True, bn=True,
                                 relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        # SubModule 的初始化
        self.weight_init()

    def forward(self, img, cost):
        """

        :param img: [x4, x8, x16, x32] xn means downsampling 1/n
        :param cost:
        :return:
        """
        b, c, h, w = img[0].shape

        cost = cost.reshape(b, -1, self.D, h, w)
        cost = self.conv_stem(cost)   # downsampling x4 head 1 --> 8
        cost = self.channelAttStem(cost, img[0])

        cost1 = self.convdown1(cost)  # downsampling x8  head 8 --> 16
        cost1 = self.channelAttdown1(cost1, img[1])

        cost2 = self.convdown2(cost1)  # downsampling x16  head 16 --> 32
        cost2 = self.channelAttdown2(cost2, img[2])

        cost3 = self.convdown3(cost2)  # downsampling x32  head 32 --> 48
        cost3 = self.channelAttdown3(cost3, img[3])

        costup = self.convup1(cost3)  #  upsampling x16  head 48 --> 32
        if costup.shape != cost2.shape:
            target_d, target_h, target_w = cost2.shape[-3:]
            costup = F.interpolate(
                costup,
                size=(target_d, target_h, target_w),
                mode='nearest')
            # ValueError: align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear
        costup = torch.cat([costup, cost2], 1)
        costup = self.convskip1(costup)  #  upsampling x16  head 32 --> 32
        costup = self.conv_agg1(costup)  #  upsampling x16  head 32 --> 32
        costup = self.channelAttup1(costup, img[2])

        costup = self.convup2(costup)  #  upsampling x8  head 32 --> 16
        if costup.shape != cost1.shape:
            target_d, target_h, target_w = cost1.shape[-3:]
            costup = F.interpolate(
                costup,
                size=(target_d, target_h, target_w),
                mode='nearest')
        costup = torch.cat([costup, cost1], 1)
        costup = self.convskip2(costup)  # upsampling x8  head 16 --> 16
        costup = self.conv_agg2(costup)  # upsampling x8  head 16 --> 16
        costup = self.channelAttup2(costup, img[1])

        costup = self.convup3(costup)  # upsampling x4  head 16 --> 1
        if costup.shape != cost.shape:
            target_d, target_h, target_w = cost.shape[-3:]
            costup = F.interpolate(
                costup,
                size=(target_d, target_h, target_w),
                mode='nearest')

        return costup


class GCE_Original(SubModule):
    def __init__(self,
                 backbone_cfg,
                 max_disparity=192,
                 matching_head=1,
                 im_channels=None,
                 channels=[16, 32, 48],
                 disp_strides=2,
                 blocks_num=[2, 2, 2],
                 gce=True,
                 ):
        super(GCE_Original, self).__init__()
        if im_channels is None:
            Log.error('{} feature channel number should be calculated before initializing the GCE Aggregation block'
                      .format(get_present_caller()))
            exit(1)
        ichans = im_channels
        self.D = max_disparity

        self.conv_stem = BasicConv(matching_head, 8,
                                   is_3d=True, kernel_size=3, stride=1, padding=1)

        self.gce = gce
        if gce:
            self.channelAttStem = ChannelAtt(8, ichans[1], self.D,)
            self.channelAtt = nn.ModuleList()
            self.channelAttDown = nn.ModuleList()

        self.conv_down = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.conv_skip = nn.ModuleList()
        self.conv_agg = nn.ModuleList()

        channels = [8] + (channels)

        s_disp = disp_strides
        block_n = blocks_num
        inp = channels[0]
        for i in range(3):
            conv = nn.ModuleList()
            for n in range(block_n[i]):
                stride = (s_disp, 2, 2) if n == 0 else 1
                dilation, kernel_size, padding, bn, relu = 1, 3, 1, True, True
                conv.append(
                    BasicConv(
                        inp, channels[i + 1], is_3d=True, bn=bn,
                        relu=relu, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation))
                inp = channels[i + 1]
            self.conv_down.append(nn.Sequential(*conv))

            if gce:
                self.channelAttDown.append(ChannelAtt(channels[i + 1],
                                                      ichans[i+2],
                                                      self.D // (2 ** (i + 1)),
                                                      ))

            if i == 0:
                out_chan, bn, relu = 1, False, False
            else:
                out_chan, bn, relu = channels[i], True, True

            if i != 0:
                self.conv_up.append(
                    BasicConv(
                        channels[i + 1], out_chan, deconv=True, is_3d=True, bn=bn,
                        relu=relu, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(s_disp, 2, 2)))

            if i != 0:
                self.conv_agg.append(nn.Sequential(
                    BasicConv(
                    channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1),
                    BasicConv(
                    channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1), ))

            if i != 0:
                self.conv_skip.append(BasicConv(2 * channels[i], channels[i], is_3d=True,
                                                kernel_size=1, padding=0, stride=1))

            if gce and i != 0:
                self.channelAtt.append(ChannelAtt(channels[i], ichans[i+1], self.D // (2 ** (i)),))

        self.convup3 = BasicConv(16, 1, deconv=True, is_3d=True, bn=True,
                                 relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.weight_init()

    def forward(self, img, cost):
        b, c, h, w = img[0].shape

        cost = cost.reshape(b, -1, self.D, h, w)
        cost = self.conv_stem(cost)
        if self.gce:
            cost = self.channelAttStem(cost, img[0])

        cost_feat = [cost]

        cost_up = cost
        for i in range(3):
            cost_ = self.conv_down[i](cost_up)
            if self.gce:
                cost_ = self.channelAttDown[i](cost_, img[i + 1])

            cost_feat.append(cost_)
            cost_up = cost_

        cost_ = cost_feat[-1]

        costup = self.conv_up[- 1](cost_)
        if costup.shape != cost_feat[- 2].shape:
            target_d, target_h, target_w = cost_feat[- 2].shape[-3:]
            costup = F.interpolate(
                costup,
                size=(target_d, target_h, target_w),
                mode='nearest')

        costup = torch.cat([costup, cost_feat[- 2]], 1)
        costup = self.conv_skip[- 1](costup)
        cost_ = self.conv_agg[- 1](costup)
        if self.gce:
            cost_ = self.channelAtt[1](cost_, img[-2])

        costup = self.conv_up[-2](cost_)
        if costup.shape != cost_feat[-3].shape:
            target_d, target_h, target_w = cost_feat[-3].shape[-3:]
            costup = F.interpolate(
                costup,
                size=(target_d, target_h, target_w),
                mode='nearest')

        costup = torch.cat([costup, cost_feat[-3]], 1)
        costup = self.conv_skip[-2](costup)
        cost_ = self.conv_agg[-2](costup)

        cost1 = self.channelAtt[-2](cost_, img[-3])

        """for i in range(2):
            if i == 2:
                break
            costup = self.conv_up[-i - 1](cost_)
            if costup.shape != cost_feat[-i - 2].shape:
                target_d, target_h, target_w = cost_feat[-i - 2].shape[-3:]
                costup = F.interpolate(
                    costup,
                    size=(target_d, target_h, target_w),
                    mode='nearest')

            costup = torch.cat([costup, cost_feat[-i - 2]], 1)
            costup = self.conv_skip[-i - 1](costup)
            cost_ = self.conv_agg[-i - 1](costup)

            if self.gce:
                cost_ = self.channelAtt[-i - 1](cost_, img[-i - 2])"""

        # 这里一开始不是这样的 但是不知道为什么就是不能运行 然后疯狂换名字

        cost2 = self.convup3(cost1)
        if cost2.shape != cost_feat[-4].shape:
            target_d, target_h, target_w = cost_feat[-4].shape[-3:]
            cost2 = F.interpolate(
                cost2,
                size=(target_d, target_h, target_w),
                mode='nearest')

        return cost2







