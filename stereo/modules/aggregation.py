import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from utils.args_dictionary import get_key, key_exist, get_present_caller
from utils.logger import Logger as Log

from stereo.modules.submodule import BasicConv, ChannelAtt, SubModule, \
    same_padding_conv, ResBlock, disp_up, hyp_up, TileExpand, SlantKeepUpsample, SlantChangeUpsample
from stereo.modules.volumehelper import DiffVolumeV2

"""
CoEx adopts the weight initialization in GCE aggregation,
but ACVFast or plus version use no weight initialization
"""

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
        # SubModule initialization
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


######################################################################
# ********************* utilization in ACV  **********************
# Accurate and Efﬁcient Stereo Matching via Attention Concatenation Volume
#
# ####################################################################

class Att_HourglassFit(SubModule):
    def __init__(self, in_channels, img_channels, final_one=False, is_cat=True):
        """

        :param in_channels:
        :param img_channels:
        :param final_one: if is true, channel of final output is set to 1
        """
        super(Att_HourglassFit, self).__init__()
        self.is_cat = is_cat

        self.hrglass_length = len(img_channels)
        self.conv_down = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.conv_agg = nn.ModuleList()
        self.channelAttDown = nn.ModuleList()
        self.channelAttUp = nn.ModuleList()

        for i in range(self.hrglass_length):
            k1 = i*2 if i > 0 else 1
            k2 = (i+1)*2
            self.conv_down.append(
                nn.Sequential(
                    BasicConv(in_channels * k1, in_channels * k2, is_3d=True, bn=True, relu=True, kernel_size=3,
                              padding=1, stride=2, dilation=1),
                    BasicConv(in_channels * k2, in_channels * k2, is_3d=True, bn=True, relu=True, kernel_size=3,
                              padding=1, stride=1, dilation=1)
            ))
            self.channelAttDown.append(ChannelAtt(in_channels*k2, img_channels[i]))

            if i == 0 and final_one:
                out_channels = 1
                bn_set, relu_set = False, False
            else:
                out_channels = in_channels * k1
                bn_set, relu_set = True, True
            self.conv_up.append(
                BasicConv(in_channels * k2, out_channels, deconv=True, is_3d=True, bn=bn_set,
                          relu=relu_set, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
            )
            if i < self.hrglass_length-1:
                mul = 2 if is_cat else 1
                self.conv_agg.append(
                    nn.Sequential(
                        BasicConv(in_channels * k2 * mul, in_channels * k2, is_3d=True, kernel_size=1, padding=0, stride=1),
                        BasicConv(in_channels * k2, in_channels * k2, is_3d=True, kernel_size=3, padding=1, stride=1),
                        BasicConv(in_channels * k2, in_channels * k2, is_3d=True, kernel_size=3, padding=1, stride=1),
                    )
                )
                self.channelAttUp.append((ChannelAtt(in_channels*k2, img_channels[i])))

        self.weight_init()

    def forward(self, volume, fea):
        assert len(fea) == self.hrglass_length

        volume_list = [volume]
        volume_down = volume

        for i in range(self.hrglass_length):
            conv_down = self.conv_down[i](volume_down)
            conv_down = self.channelAttDown[i](conv_down, fea[i])
            volume_list.append(conv_down)
            volume_down = conv_down

        volume_up = volume_list[-1]

        for i in range(self.hrglass_length-1):
            conv_up = self.conv_up[-i-1](volume_up)
            if self.is_cat:
                conv_up = torch.cat((conv_up, volume_list[-i-2]), dim=1)
            else:
                conv_up = conv_up + volume_list[-i-2]
            conv_up = self.conv_agg[-i-1](conv_up)
            volume_up = self.channelAttUp[-i-1](conv_up, fea[-i-2])

        volume_final = self.conv_up[0](volume_up)
        return volume_final


######################################################################
# ********************* utilization in CoEx  **********************
# Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation
#
# ####################################################################

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
            self.channelAttStem = ChannelAtt(8, ichans[1])
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

        cost2 = self.convup3(cost1)
        if cost2.shape != cost_feat[-4].shape:
            target_d, target_h, target_w = cost_feat[-4].shape[-3:]
            cost2 = F.interpolate(
                cost2,
                size=(target_d, target_h, target_w),
                mode='nearest')

        return cost2


class GCE(SubModule):
    # structure details in CoEx
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


######################################################################
# ********************* utilization in HITNet  **********************
# HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching
# https://github.com/zjjMaiMai/TinyHITNet
# ####################################################################


class Tile_INIT(nn.Module):
    def __init__(self, in_channels, max_disp, coarsefea=16):
        super(Tile_INIT, self).__init__()
        self.max_disp = max_disp
        self.des_channel = 13  # Tile descriptor has 13 channels by default

        # To extract the tile features we run a 4 × 4 convolution on each extracted feature map
        self.conv_fea_x = nn.Conv2d(in_channels, 16, kernel_size=4)  # i6?

        self.conv_mth_x = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 1),
            nn.LeakyReLU(0.2),
        )  # This convolution is followed by a leaky ReLU and an MLP

        self.conv_des_x = nn.Sequential(
            nn.Conv2d(coarsefea + 1, self.des_channel, kernel_size=1),
            nn.LeakyReLU(0.2),
        )

        self.volume = DiffVolumeV2(None)

    def forward(self, fea_l, fea_r, coarse=None):
        match_l = F.conv2d(
            fea_l,
            self.conv_fea_x.weight,
            self.conv_fea_x.bias,
            stride=(4, 4),
        ) # For the left image we use strides of 4 × 4
        match_l = self.conv_mth_x(match_l)

        match_r = same_padding_conv(
            fea_r,
            self.conv_fea_x.weight,
            self.conv_fea_x.bias,
            s=(4, 1),
        )
        # the right image we use strides of 4 × 1, which is crucial to
        # maintain the full disparity resolution to maximize accuracy
        match_r = self.conv_mth_x(match_r)

        volume = self.volume(match_l, match_r, self.max_disp)
        # print(match_r.size())
        sad_volume = torch.norm(volume, p=1, dim=1)
        # print(sad_volume.size())
        # Eq 3:
        cost_min, d_init = torch.min(sad_volume, dim=1, keepdim=True)
        d_init = d_init.float()

        if coarse is None:
            coarse = match_l

        p = torch.cat((cost_min, coarse), dim=1)
        # page 18 Fig.19 use the coarse resolution if available
        descriptor = self.conv_des_x(p)  # 13

        hyp_int = torch.cat((d_init, torch.zeros_like(d_init), torch.zeros_like(d_init), descriptor), dim=1)

        return hyp_int, sad_volume


def augmented_hypothesis(disp, fea_l, fea_r, local):
    b, c, h, w = fea_r.size()
    x_index = torch.arange(w, device=fea_l.device)
    y_index = torch.arange(h, device=fea_l.device)
    coef_y, coef_x = torch.meshgrid(y_index, x_index, indexing="ij")
    coef_y = coef_y.view(1, 1, h, w).repeat(b, 1, 1, 1)
    coef_x = coef_x.view(1, 1, h, w).repeat(b, 1, 1, 1)
    coef_dx = coef_x - disp

    coef_dx = coef_dx.permute(0, 2, 3, 1)
    coef_y = coef_y.permute(0, 2, 3, 1)
    coef_y = 2.0 * coef_y / max(h - 1, 1) - 1.0
    w_scale = max(w - 1, 1)

    aug_hyp = [torch.sum(torch.abs(fea_l), dim=1, keepdim=True)]
    for offset in range(-1 * local, local + 1, 1):
        x_dex = coef_dx.clone() + offset
        x_dex = torch.clip(x_dex, min=0, max=w - 1)
        x_dex = 2.0 * x_dex / w_scale - 1.0
        vgrid = torch.cat((x_dex, coef_y), dim=3).float()
        warp_right = F.grid_sample(fea_r, vgrid)
        aug_hyp.append(torch.sum(torch.abs(fea_l - warp_right), dim=1, keepdim=True))

    aug_hyp = torch.cat(aug_hyp, dim=1)  # [B, 2local+1, H, W]

    return aug_hyp


class Mid_Prop(nn.Module):
    def __init__(self, hyp_num=2, res_num=2, dilation=[1, 1]):
        super(Mid_Prop, self).__init__()
        tile_size = 4
        patch_size = tile_size * tile_size  # 16
        self.hyp_num = hyp_num
        self.local = 1
        aug_hyp = 64  # (self.local * 2 + 1 + 1) * patch_size

        if hyp_num == 2:
            # 为什么只有2的时候才有
            self.dispup = SlantKeepUpsample(2)
            self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        # Before running a sequence of residual blocks with varying dilation factors we run a 1×1 convolution followed
        # by a leaky ReLU to decrease the number of feature channels.
        self.conv_neighbors = nn.Sequential(
            nn.Conv2d((aug_hyp + 16) * self.hyp_num, 32, 1),
            nn.LeakyReLU(0.2),
        )  # decrease the num

        # resblock use 32 channels by default
        res: List[nn.Module] = []
        # Page 15: res block use 32 channels
        # intermediate propagation steps use 2 res block without dilation
        for i in range(res_num):
            res.append(ResBlock(32, dilation[i]))

        self.res = nn.Sequential(*res)
        self.convn = nn.Conv2d(32, 17 * hyp_num, 3, 1, 1)
        # the details of the tile was show in Fig.12
        self.relu = nn.ReLU()

    def augmented_hypothesis(self, hyp, fea_l, fea_r):
        # page 18 Fig 10, Propagation
        # Augment tile constructed by 3 local cost and abs left feature
        scale = fea_l.size(3) // hyp.size(3)
        assert scale == 4  # tile_size

        tile_expand = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile=True)  # [b, 1, h * scale, w  * scale]

        b, c, h, w = fea_r.size()
        x_index = torch.arange(w, device=fea_l.device)
        y_index = torch.arange(h, device=fea_l.device)
        coef_y, coef_x = torch.meshgrid(y_index, x_index, indexing="ij")
        coef_y = coef_y.view(1, 1, h, w).repeat(b, 1, 1, 1)
        coef_x = coef_x.view(1, 1, h, w).repeat(b, 1, 1, 1)
        coef_dx = coef_x - tile_expand

        coef_dx = coef_dx.permute(0, 2, 3, 1)
        coef_y = coef_y.permute(0, 2, 3, 1)
        coef_y = 2.0 * coef_y / max(h - 1, 1) - 1.0
        w_scale = max(w - 1, 1)

        aug_hyp = [torch.sum(torch.abs(fea_l), dim=1, keepdim=True)]  # ads tile_size * tile_size larger than the original hyp
        for offset in range(-1 * self.local, self.local + 1, 1):
            x_dex = coef_dx.clone() + offset
            x_dex = torch.clip(x_dex, min=0, max=w - 1)
            x_dex = 2.0 * x_dex / w_scale - 1.0
            vgrid = torch.cat((x_dex, coef_y), dim=3).float()
            warp_right = F.grid_sample(fea_r, vgrid)
            aug_hyp.append(torch.sum(torch.abs(fea_l - warp_right), dim=1, keepdim=True))
            # print(warp_right.size(), "-----------------------")
            # print(aug_hyp[-1].size())

        aug_hyp = torch.cat(aug_hyp, dim=1)
        b, c, h, w = aug_hyp.size()
        aug_hyp = aug_hyp.reshape(b, c, h // scale, scale, w // scale, scale)
        aug_hyp = aug_hyp.permute(0, 3, 5, 1, 2, 4)
        aug_hyp = aug_hyp.reshape(b, scale * scale * c, h // scale, w // scale)
        return aug_hyp

    def forward(self, hyp, fea_l, fea_r, hyp_pre=None):
        if self.hyp_num == 2 and hyp_pre is None:
            Log.error("Wrong!")
            exit(1)
        aug_hyp = self.augmented_hypothesis(hyp, fea_l, fea_r)
        if hyp_pre is None:
            aug_hyp_set = torch.cat([aug_hyp, hyp], dim=1)
            hyp_set = hyp
        elif hyp_pre is not None:
            # page 5:
            # Thereby, the disparity d is upsampled using the plane equation of the tile and the remaining
            # parts of the tile hypothesis dx, dy and p are upsampled using nearest neighbor sampling.
            d_pre = hyp_pre[:, 0, :, :].unsqueeze(1)  # multiply 2 when passing to slant upsampling
            dx_pre = hyp_pre[:, 1, :, :].unsqueeze(1)  # h direction
            dy_pre = hyp_pre[:, 2, :, :].unsqueeze(1)  # w direction
            d_up = self.dispup(d_pre, dx_pre, dy_pre)
            dxy_up = self.upsample(hyp_pre[:, 1:3, :, :])
            dscrpt_up = self.upsample(hyp_pre[:, 3:, :, :])
            hyp_fine = torch.cat([d_up, dxy_up, dscrpt_up], dim=1)
            coarse_aug_hyp = self.augmented_hypothesis(hyp_fine, fea_l, fea_r)
            hyp_set = torch.cat([hyp, hyp_fine], dim=1)
            aug_hyp_set = torch.cat([aug_hyp, coarse_aug_hyp, hyp_set], dim=1)

        x = self.conv_neighbors(aug_hyp_set)
        x = self.res(x)
        x = self.convn(x)

        dh = x[:, : 16 * self.hyp_num]  # 这种索引方式就是对第二维度索引
        w = x[:, 16 * self.hyp_num:]
        # 和位置有关嘛

        hyp_refined = hyp_set + dh
        hyp_refined[:, :1, :, :] = self.relu(hyp_refined[:, :1, :, :].clone())
        hyp_refined[:, 16:17, :, :] = self.relu(hyp_refined[:, 16:17, :, :].clone())
        return hyp_refined, w


class Mid_PropV2(nn.Module):
    def __init__(self, hyp_num=2, res_num=2, dilation=[1,1]):
        super(Mid_PropV2, self).__init__()
        tile_size = 4
        patch_size = tile_size * tile_size  # 16
        self.hyp_num = hyp_num
        self.local = 1
        aug_hyp = 64  # (self.local * 2 + 1 + 1) * patch_size
        self.conv_neighbors = nn.Sequential(
            nn.Conv2d(aug_hyp * self.hyp_num, patch_size * hyp_num, 1),
            nn.LeakyReLU(0.2),
        )  # decrease the num

        # See Fig.12 the
        self.conv1 = nn.Sequential(
            nn.Conv2d(32 * hyp_num, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        # resblock use 32 channels by default
        res: List[nn.Module] = []
        # Page 15: res block use 32 channels
        # intermediate propagation steps use 2 res block without dilation
        for i in range(res_num):
            res.append(ResBlock(32, dilation[i]))

        self.res = nn.Sequential(*res)
        self.convn = nn.Conv2d(32, 17 * hyp_num, 3, 1, 1)
        # the details of the tile was show in Fig.12

    def augmented_hypothesis(self, hyp, fea_l, fea_r):
        # page 18 Fig 10, Propagation
        # Augment tile constructed by 3 local cost and abs left feature
        scale = fea_l.size(3) // hyp.size(3)
        assert scale == 4  # tile_size

        tile_expand = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile=True)  # [b, 1, h * scale, w  * scale]

        b, c, h, w = fea_r.size()
        x_index = torch.arange(w, device=fea_l.device)
        y_index = torch.arange(h, device=fea_l.device)
        coef_y, coef_x = torch.meshgrid(y_index, x_index, indexing="ij")
        coef_y = coef_y.view(1, 1, h, w).repeat(b, 1, 1, 1)
        coef_x = coef_x.view(1, 1, h, w).repeat(b, 1, 1, 1)
        coef_dx = coef_x - tile_expand

        coef_dx = coef_dx.permute(0, 2, 3, 1)
        coef_y = coef_y.permute(0, 2, 3, 1)
        coef_y = 2.0 * coef_y / max(h - 1, 1) - 1.0
        w_scale = max(w - 1, 1)

        aug_hyp = [torch.sum(torch.abs(fea_l), dim=1, keepdim=True)]  # ads tile_size * tile_size larger than the original hyp
        for offset in range(-1 * self.local, self.local + 1, 1):
            x_dex = coef_dx.clone() + offset
            x_dex = torch.clip(x_dex, min=0, max=w - 1)
            x_dex = 2.0 * x_dex / w_scale - 1.0
            vgrid = torch.cat((x_dex, coef_y), dim=3).float()
            warp_right = F.grid_sample(fea_r, vgrid)
            aug_hyp.append(torch.sum(torch.abs(fea_l - warp_right), dim=1, keepdim=True))

        aug_hyp = torch.cat(aug_hyp, dim=1)
        b, c, h, w = aug_hyp.size()
        aug_hyp = aug_hyp.reshape(b, c, h // scale, scale, w // scale, scale)
        aug_hyp = aug_hyp.permute(0, 3, 5, 1, 2, 4)
        aug_hyp = aug_hyp.reshape(b, scale * scale * c, h // scale, w // scale)
        return aug_hyp

    def forward(self, hyps, fea_l, fea_r):
        aug_hyp = [self.augmented_hypothesis(h, fea_l, fea_r) for h in hyps]
        aug_hyp = torch.cat(aug_hyp, dim=1)
        x = self.conv_neighbors(aug_hyp)
        hyps = torch.cat(hyps, dim=1)
        x = torch.cat((hyps, x), dim=1)

        x = self.conv1(x)
        x = self.res_block(x)
        x = self.convn(x)

        dh = x[:, : 16 * self.h_size]  # 这种索引方式就是对第二维度索引
        w = x[:, 16 * self.h_size:]
        # 和位置有关嘛
        return hyps + dh, w


class Last_Prop(nn.Module):
    # To achieve that, they operate
    # on coarse feature maps: the 4×4 tiles use 4X downsampled
    # features for warping, the 2 × 2 tiles use 2X downsampled
    # features for warping, the 1 × 1 tiles use full-resolution features for warping.
    def __init__(self, fea_chan=2, res_num=2, res_channel=32, dilation=[1, 1], up_function=None, upscale=None, final=False):
        super(Last_Prop, self).__init__()
        self.disp_upsample = up_function
        self.local = 1
        basic_local=4
        self.final = final
        if up_function is not None and upscale is None:
            Log.error("upscale should be not None when using a up_function")
            exit(1)
        if up_function is not None:
            self.upsample = nn.UpsamplingNearest2d(scale_factor=upscale)

        # Before running a sequence of residual blocks with varying dilation factors we run a 1×1 convolution followed
        # by a leaky ReLU to decrease the number of feature channels.
        self.conv_neighbors = nn.Sequential(
            nn.Conv2d((4 + 16), res_channel, 1),
            nn.LeakyReLU(0.2),
        )  # decrease the num

        assert res_num == len(dilation)
        res: List[nn.Module] = []
        for i in range(res_num):
            res.append(ResBlock(res_channel, dilation[i]))

        self.res = nn.Sequential(*res)
        if final:
            outc = 1
        elif not final:
            outc = 16
        self.convn = nn.Conv2d(res_channel, outc, 3, 1, 1)

        self.relu = nn.ReLU()

    def forward(self, hyp, fea_l, fea_r):
        if self.disp_upsample is not None:
            d_prehyp = hyp[:, 0, :, :].unsqueeze(1)
            dx_prehyp = hyp[:, 1, :, :].unsqueeze(1)
            dy_prehyp = hyp[:, 2, :, :].unsqueeze(1)
            d_up = self.disp_upsample(d_prehyp, dx_prehyp, dy_prehyp)
            dx_up = self.upsample(hyp[:, 1:3, :, :])
            dscrpt_up = self.upsample(hyp[:, 3:, :, :])
            hyp = torch.cat([d_up, dx_up, dscrpt_up], dim=1)

        aug_hyp = augmented_hypothesis(hyp[:, :1], fea_l, fea_r, self.local)

        x = torch.cat([aug_hyp, hyp], dim=1)
        x = self.conv_neighbors(x)

        x = self.res(x)
        x = self.convn(x)

        if not self.final:
            hyp_refined = x + hyp
            hyp_refined[:, :1, :, :] = self.relu(hyp_refined[:, :1, :, :].clone())
        elif self.final:
            hyp_refined = self.relu(torch.squeeze(x, dim=1) + hyp[:, :1, :, :])
            hyp_refined = hyp_refined.squeeze(dim=1)
            # final output is 3dim

        return hyp_refined

class Last_PropV2(nn.Module):
    def __init__(self, fea_chan=2, res_num=2, res_channel=32, dilation=[1, 1]):
        super(Last_PropV2, self).__init__()
        # fea + 3 cost
        self.local = 1
        self.conv_neighbors = nn.Sequential(
            nn.Conv2d(16+4, res_channel, 1),
        )
        assert res_num == len(dilation)
        res: List[nn.Module] = []
        for i in range(res_num):
            res.append(ResBlock(res_channel, dilation[i]))

        self.res = nn.Sequential(*res)
        self.convn = nn.Conv2d(32, 16, 3, 1, 1)

    def augmented_hypothesis(self, disp, fea_l, fea_r):
        b, c, h, w = fea_r.size()
        x_index = torch.arange(w, device=fea_l.device)
        y_index = torch.arange(h, device=fea_l.device)
        coef_y, coef_x = torch.meshgrid(y_index, x_index, indexing="ij")
        coef_y = coef_y.view(1, 1, h, w).repeat(b, 1, 1, 1)
        coef_x = coef_x.view(1, 1, h, w).repeat(b, 1, 1, 1)
        coef_dx = coef_x - disp

        coef_dx = coef_dx.permute(0, 2, 3, 1)
        coef_y = coef_y.permute(0, 2, 3, 1)
        coef_y = 2.0 * coef_y / max(h - 1, 1) - 1.0
        w_scale = max(w - 1, 1)

        aug_hyp = [torch.sum(torch.abs(fea_l), dim=1, keepdim=True)]
        for offset in range(-1 * self.local, self.local + 1, 1):
            x_dex = coef_dx.clone() + offset
            x_dex = torch.clip(x_dex, min=0, max=w - 1)
            x_dex = 2.0 * x_dex / w_scale - 1.0
            vgrid = torch.cat((x_dex, coef_y), dim=3).float()
            warp_right = F.grid_sample(fea_r, vgrid)
            aug_hyp.append(torch.sum(torch.abs(fea_l - warp_right), dim=1, keepdim=True))

        aug_hyp = torch.cat(aug_hyp, dim=1)  # [B, 2local+1, H, W]

        return aug_hyp

    def forward(self, hyp, fea_l, fea_r, scale=1):
        if scale > 1:
            assert fea_r.size(3)/hyp.size(3) == scale
            hyp = hyp_up(hyp, 1, 2)

        aug_hyp = self.augmented_hypothesis(hyp[:, :1], fea_l, fea_r)
        aug_hyp = torch.cat((hyp, aug_hyp), dim=1)
        x = self.conv_neighbors(aug_hyp)

        x = self.res(x)
        x = self.convn(x)

        return hyp + x





