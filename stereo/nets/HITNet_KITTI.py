import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math

from utils.logger import Logger as Log
from utils.args_dictionary import get_key

from stereo.modules.submodule import disp_up, hyp_up, TileExpand, SlantChangeUpsample, SlantKeepUpsample
from stereo.modules.feature import HITNET_Feature
from stereo.modules.aggregation import Tile_INIT, Mid_Prop, Last_Prop


class HITNet_KITTI(nn.Module):
    def __init__(self, args):
        super(HITNet_KITTI, self).__init__()
        self.name = "HITNet_KITTI"
        self.align = 64

        net_cfg = get_key(args, 'net')
        hit_cfg = get_key(args, 'net', 'hit')
        backbone = get_key(args, 'net', 'backbone')
        # refinement = get_key(args, 'net', 'refinement')
        self.D = get_key(args, 'dataset', 'max_disparity')  # 256

        self.feature = HITNET_Feature(backbone)
        fea_chan = self.feature.ccount()  # [16, 16, 24, 24, 32] verse

        #  init tile size is 1/(hyp size x fea size) and the disp range is max disp / fea size
        #  for fea range have to go through a 4/1 down-sampling for left and right fea
        self.tile_init = nn.ModuleList(
            [
                Tile_INIT(fea_chan[-1], self.D // 16),
                Tile_INIT(fea_chan[-2], self.D // 8),
                Tile_INIT(fea_chan[-3], self.D // 4, coarsefea=fea_chan[-1]),
                Tile_INIT(fea_chan[-4], self.D // 2, coarsefea=fea_chan[-2]),
                Tile_INIT(fea_chan[-5], self.D // 1, coarsefea=fea_chan[-3]),
            ]
        )
        # return [d_init, 0, 0 description] # description is 13 by default
        # size (B, F, H, W) F is 16 by default, H and W is 1/tile size of the current fea map and disp research range

        self.mid_prop = nn.ModuleList(
            [
                Mid_Prop(hyp_num=1),
                Mid_Prop(hyp_num=2),
                Mid_Prop(hyp_num=2),
                Mid_Prop(hyp_num=2),
                Mid_Prop(hyp_num=2),
            ]
        )

        self.last3prop = nn.ModuleList(
            [
                Last_Prop(fea_chan=fea_chan[2], res_num=4, res_channel=32, dilation=[1, 3, 1, 1]),
                Last_Prop(fea_chan=fea_chan[1], res_num=4, res_channel=32, dilation=[1, 3, 1, 1], up_function=SlantChangeUpsample(2), upscale=2),
                Last_Prop(fea_chan=fea_chan[0], res_num=2, res_channel=16, dilation=[1, 1],
                          up_function=SlantKeepUpsample(2, 2), upscale=2, final=True),
            ]
        )

        if self.training:
            self.disp_upsample64x = TileExpand(64)
            self.disp_upsample32x = TileExpand(32)
            self.disp_upsample16x = TileExpand(16)
            self.disp_upsample8x = TileExpand(8)
            self.disp_upsample4x = TileExpand(4)
            self.disp_upsample2x = TileExpand(2, 2)
            self.near_upsample64x = nn.UpsamplingNearest2d(scale_factor=64)
            self.near_upsample32x = nn.UpsamplingNearest2d(scale_factor=32)
            self.near_upsample16x = nn.UpsamplingNearest2d(scale_factor=16)
            self.near_upsample8x = nn.UpsamplingNearest2d(scale_factor=8)
            self.near_upsample4x = nn.UpsamplingNearest2d(scale_factor=4)
            self.near_upsample2x = nn.UpsamplingNearest2d(scale_factor=2)

    def hyp_select(self, h, w):
        h0 = h[:, :16]
        h1 = h[:, 16:]
        w0 = w[:, :1]
        w1 = w[:, 1:]
        h = torch.where(w0 > w1, h0, h1)
        return h, [[w0, h0[:, :1]], [w1, h1[:, :1]]]

    def forward(self, left_img, right_img):
        b, c, h, w = left_img.size()

        cat = torch.cat((left_img, right_img), dim=0)

        f_list = self.feature(cat)

        fl, fr = [], []  #  from 16x to 1x
        for v_ in f_list:
            fl_, fr_ = v_.split(dim=0, split_size=b)
            fl.append(fl_)
            fr.append(fr_)

        # Initialization
        hyp_0, volume0 = self.tile_init[0](fl[0], fr[0])
        hyp_1, volume1 = self.tile_init[1](fl[1], fr[1])
        hyp_2, volume2 = self.tile_init[2](fl[2], fr[2], coarse=fl[0])
        hyp_3, volume3 = self.tile_init[3](fl[3], fr[3], coarse=fl[1])
        hyp_4, volume4 = self.tile_init[4](fl[4], fr[4], coarse=fl[2])

        if self.training:
            volume_pyramid = [volume0, volume1, volume2, volume3, volume4]  # used for initial loss
            conf_pyramid = []
            slant_pyramid = []
            preds_pyramid = []
            conf_pyramid_coarse = []
            slant_pyramid_coarse = []
            preds_pyramid_coarse = []
        del volume0, volume1, volume2, volume3, volume4

        hyp_0, wp0 = self.mid_prop[0](hyp_0, fl[0], fr[0])
        if self.training:
            preds_pyramid.append(self.disp_upsample64x(hyp_0[:, :1], hyp_0[:, 1:2], hyp_0[:, 2:3], squeeze_dim=1))  # disp 16x
            slant_pyramid.append(self.near_upsample64x(hyp_0[:, 1:3]))  # 16x

        hyp, w = self.mid_prop[1](hyp_1, fl[1], fr[1], hyp_pre=hyp_0)
        if self.training:
            # hyp, hyp_pre
            preds_pyramid.append(self.disp_upsample32x(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], squeeze_dim=1))  # disp 8x cur hyp
            preds_pyramid_coarse.append(self.disp_upsample32x(hyp[:, 16:17], hyp[:, 17:18], hyp[:, 18:19], squeeze_dim=1))  # disp 8x pre hyp
            slant_pyramid.append(self.near_upsample32x(hyp[:, 1:3])) # slant 8x cur hyp
            slant_pyramid_coarse.append(self.near_upsample32x(hyp[:, 17:19]))  # slant 8x pre hyp
            conf_pyramid.append(self.near_upsample32x(w[:, :1]))  # cur
            conf_pyramid_coarse.append(self.near_upsample32x(w[:, 1:]))  # pre

        hyp_1, wp1 = self.hyp_select(hyp, w)
        hyp, w = self.mid_prop[2](hyp_2, fl[2], fr[2], hyp_pre=hyp_1)
        if self.training:
            # hyp, hyp_pre
            preds_pyramid.append(self.disp_upsample16x(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], squeeze_dim=1))  # disp 8x cur hyp
            preds_pyramid_coarse.append(self.disp_upsample16x(hyp[:, 16:17], hyp[:, 17:18], hyp[:, 18:19], squeeze_dim=1))  # disp 8x pre hyp
            slant_pyramid.append(self.near_upsample16x(hyp[:, 1:3])) # slant 8x cur hyp
            slant_pyramid_coarse.append(self.near_upsample16x(hyp[:, 17:19]))  # slant 8x pre hyp
            conf_pyramid.append(self.near_upsample16x(w[:, :1]))
            conf_pyramid_coarse.append(self.near_upsample16x(w[:, 1:]))

        hyp_2, wp2 = self.hyp_select(hyp, w)
        hyp, w = self.mid_prop[3](hyp_3, fl[3], fr[3], hyp_2)
        if self.training:
            # hyp, hyp_pre
            preds_pyramid.append(self.disp_upsample8x(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], squeeze_dim=1))  # disp 8x cur hyp
            preds_pyramid_coarse.append(self.disp_upsample8x(hyp[:, 16:17], hyp[:, 17:18], hyp[:, 18:19], squeeze_dim=1))  # disp 8x pre hyp
            slant_pyramid.append(self.near_upsample8x(hyp[:, 1:3]))  # slant 8x cur hyp
            slant_pyramid_coarse.append(self.near_upsample8x(hyp[:, 17:19]))  # slant 8x pre hyp
            conf_pyramid.append(self.near_upsample8x(w[:, :1]))
            conf_pyramid_coarse.append(self.near_upsample8x(w[:, 1:]))

        hyp_3, wp3 = self.hyp_select(hyp, w)
        hyp, w = self.mid_prop[4](hyp_4, fl[4], fr[4], hyp_pre=hyp_3)
        if self.training:
            # hyp, hyp_pre
            preds_pyramid.append(self.disp_upsample4x(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], squeeze_dim=1))  # disp 8x cur hyp
            preds_pyramid_coarse.append(self.disp_upsample4x(hyp[:, 16:17], hyp[:, 17:18], hyp[:, 18:19], squeeze_dim=1))  # disp 8x pre hyp
            slant_pyramid.append(self.near_upsample4x(hyp[:, 1:3]))  # slant 8x cur hyp
            slant_pyramid_coarse.append(self.near_upsample4x(hyp[:, 17:19]))  # slant 8x pre hyp
            conf_pyramid.append(self.near_upsample4x(w[:, :1]))
            conf_pyramid_coarse.append(self.near_upsample4x(w[:, 1:]))

        hyp_4, wp4 = self.hyp_select(hyp, w)

        hyp_5 = self.last3prop[0](hyp_4, fl[-3], fr[-3])
        # To further refine the disparity
        # map we use the winning hypothesis for the 4 × 4 tiles and
        # apply propagation module 3 times: for 4 × 4, 2 × 2, 1 × 1 resolutions
        if self.training:
            # hyp, hyp_pre
            preds_pyramid.append(self.disp_upsample4x(hyp_5[:, :1], hyp_5[:, 1:2], hyp_5[:, 2:3], squeeze_dim=1))  # disp 8x cur hyp
            slant_pyramid.append(self.near_upsample4x(hyp_5[:, 1:3]))  # slant 8x cur hyp

        hyp_6 = self.last3prop[1](hyp_5, fl[-2], fr[-2])
        if self.training:
            # hyp, hyp_pre
            preds_pyramid.append(self.disp_upsample2x(hyp_6[:, :1], hyp_6[:, 1:2], hyp_6[:, 2:3], squeeze_dim=1))  # disp 8x cur hyp
            slant_pyramid.append(self.near_upsample2x(hyp_6[:, 1:3]))  # slant 8x cur hyp

        hyp_7 = self.last3prop[2](hyp_6, fl[-1], fr[-1])

        if self.training:
            preds_pyramid.append(hyp_7)
            # return preds_pyramid, slant_pyramid, conf_pyramid, volume_pyramid
            return {
                "preds_pyramid": preds_pyramid,
                "preds_pyramid_coarse": preds_pyramid_coarse,
                "slant_pyramid": slant_pyramid,
                "slant_pyramid_coarse": slant_pyramid_coarse,
                "confidence_pyramid": conf_pyramid,
                "confidence_pyramid_coarse": conf_pyramid_coarse,
                "volume_pyramid": volume_pyramid,
            }

        return {
            "preds_pyramid": [hyp_7],
        }

    # @staticmethod
    def get_name(self):
        return self.name

if __name__ == "__main__":
    # import cv2

    args={
        "net":
            {
                "hit":{
                    "descriptor": 13,
                    "propogation": [2, 2, 2, 4, 4, 2],
                    "dilations": [[1, 1], [1, 1], [1, 1], [1, 3, 1, 1], [1, 3, 1, 1], [1, 1]],
                },
                "backbone":
                    {
                        "type": "HITNetKITTI",
                        "channels": [16, 16, 24, 24, 32],
                    },
             },
        "dataset":
            {
                "max_disparity": 256,
            },
    }

    left = torch.rand(1, 3, 320, 1152)
    right = torch.rand(1, 3, 320, 1152)
    model = HITNet_KITTI(args)

    print(model(left, right)["preds_pyramid"][-1].size())




