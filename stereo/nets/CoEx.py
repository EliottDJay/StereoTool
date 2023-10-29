from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger as Log
from utils.args_dictionary import get_key

from stereo.modules.submodule import BasicConv, Conv2x
from stereo.modules.feature import BasicFeature, BasicFeaUp
from stereo.modules.volumehelper import VolumeHelper
from stereo.modules.aggregation import GCE
from stereo.modules.regression import regression_topk
from stereo.modules.spixel import upfeat

class CoEx(nn.Module):
    def __init__(self, args):
        # https://blog.csdn.net/dongjinkun/article/details/117232330
        super(CoEx, self).__init__()
        # self.cfg = args
        self.name = "CoEx"
        self.backbone = get_key(args, 'net', 'backbone')
        self.refinement = get_key(args, 'net', 'refinement')
        self.D = get_key(args, 'dataset', 'max_disparity')

        # feature extracter mobilenetv2 [16,24,32,96,160]
        self.feature = BasicFeature(self.backbone)
        self.up = BasicFeaUp(self.backbone)
        originalc = self.backbone['channels']
        fea_c, fea_c_up = self.up.ccount()
        #if self.refinement['name'] == 'spixel': 没必要 这里就是使用的 spx
        spxc = self.refinement['spxc']

        fea_c_up[1] = fea_c_up[1] + spxc[1]
        # 这里是在feature之后进行spx的特征的提取，所以会需要在这个基础上进行相加
        volumeset = get_key(args, 'net', "volume1")
        self.volume_construction = VolumeHelper(volumeset)

        self.aggregation = GCE(args, self.D//4, matching_head=volumeset['group'], im_channels=fea_c_up)

        # spx
        self.stem_x2 = nn.Sequential(
            nn.Conv2d(3, spxc[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(spxc[0]),
            nn.LeakyReLU(),
            nn.Conv2d(spxc[0], spxc[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[0]),
            nn.LeakyReLU()  # 原代码使用的是的 ReLU
        )
        self.stem_x4 = nn.Sequential(
            nn.Conv2d(spxc[0], spxc[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(spxc[1]),
            nn.LeakyReLU(),
            nn.Conv2d(spxc[1], spxc[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[1]),
            nn.LeakyReLU()  # 原代码使用的是的 ReLU
        )

        self.spx_4x = nn.Sequential(
            BasicConv(fea_c_up[1], fea_c[1], kernel_size=3, stride=1, padding=1), # bn and leakyrelu
            nn.Conv2d(fea_c[1], fea_c[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(fea_c[1]),
            nn.ReLU()
        )
        self.spx_2x = Conv2x(fea_c[1], spxc[0], True) # Concat  spxc[0] --> 32
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*spxc[0], 9, kernel_size=4, stride=2, padding=1))

    def forward(self, left_img, right_img):
        b, c, h, w = left_img.shape
        disp_pred = []

        cat = torch.cat((left_img, right_img), dim=0)
        f_x2, f_list = self.feature(cat)

        fl_x2, _ = f_x2.split(dim=0, split_size=b)
        f_list = self.up(f_list)

        # spx feature
        spxf_x2 = self.stem_x2(cat)
        spxf_x4 = self.stem_x4(spxf_x2)
        spxfl_x2, _ = spxf_x2.split(dim=0, split_size=b)
        spxfl_x4, spxfr_x4 = spxf_x4.split(dim=0, split_size=b)

        # for CoEx it may be no mean
        fl, fr = [], []
        for v_ in f_list:
            fl_, fr_ = v_.split(dim=0, split_size=b)
            fl.append(fl_)
            fr.append(fr_)

        fl[0] = torch.cat((fl[0], spxfl_x4), 1)
        fr[0] = torch.cat((fr[0], spxfr_x4), 1)

        # cost volume
        volume = self.volume_construction(fl[0], fr[0], self.D // 4)
        volume = self.aggregation(fl, volume)

        # spx
        xspx = self.spx_4x(fr[0])
        xspx = self.spx_2x(xspx, spxfl_x2)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        # Regression
        volume = torch.squeeze(volume, 1)
        # regression_topk only allow 3D volume input
        disp_pred_x4 = regression_topk(volume, 3, self.D // 4)  # 4D volume
        if self.training:
            disp_pred.append(disp_pred_x4)
        disp_pred_x1 = upfeat(disp_pred_x4.unsqueeze(dim=1), spx_pred)
        disp_pred.append(disp_pred_x1.squeeze(dim=1)*4)
        # 为什么原论文要append 0
        # make sure disp_pred is (B, W, H)
        # dont forget *4

        # we should keep sure that output is all list or dir
        return disp_pred

    # @staticmethod
    def get_name(self):
        return self.name












