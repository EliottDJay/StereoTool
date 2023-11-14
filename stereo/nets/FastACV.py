from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import Logger as Log
from utils.args_dictionary import get_key

from stereo.modules.submodule import BasicConv, Conv2x, ChannelAtt, disparity_variance, Propagation, SpatialTransformer_grid, Propagation_prob
from stereo.modules.feature import BasicFeature, BasicFeaUp
from stereo.modules.volumehelper import VolumeHelper, SparseConcatVolume
from stereo.modules.aggregation import Att_HourglassFit
from stereo.modules.regression import regression_topk_sparse, disparity_regression
from stereo.modules.spixel import upfeat


class FastACV_Original(nn.Module):
    def __init__(self, args):
        super(FastACV_Original, self).__init__()

        self.name = "FastACV_Original"
        net_cfg = get_key(args, 'net')
        acv_cfg = get_key(args, 'net', 'acv')
        backbone = get_key(args, 'net', 'backbone')
        refinement = get_key(args, 'net', 'refinement')
        self.D = get_key(args, 'dataset', 'max_disparity')
        if "regression" in net_cfg.keys():
            if "topk" in net_cfg["regression"].keys():
                self.topk = net_cfg["regression"]["topk"]
        else:
            self.topk = 3

        ####################################################################
        # ************** Feature Structure and Channel Number *************
        ####################################################################
        self.feature = BasicFeature(backbone)  # Feature Extractor Mobilenetv2 [16,24,32,96,160]
        self.up = BasicFeaUp(backbone)
        originalc = backbone['channels']  # Original Feature Extractor Feature Channel Number
        fea_c, fea_c_up = self.up.ccount()
        # if self.refinement['name'] == 'spixel'  # Needless, spx used in original net
        spxc = refinement['spxc']
        fea_c_up[1] = fea_c_up[1] + spxc[1]
        fea_c_up[2] = fea_c_up[2] + spxc[2]

        ####################################################################
        # ********* Corr Volume and Concatenation Volume Structure  ********
        ####################################################################
        # ACV
        self.coarse_topk = acv_cfg["coarse_topk"]
        self.attention_weight_only = acv_cfg["att_weights_only"]  # top k value in the coarse map
        print(self.attention_weight_only)
        # Volume details
        att_volume_set = get_key(args, 'net', "volume1")
        cat_volume_set = get_key(args, 'net', "volume2")
        self.corr_volume_construction = VolumeHelper(att_volume_set)
        group = att_volume_set["group"]
        cat_group = cat_volume_set["group"]  # 32
        # Corr Volume aggregation
        self.patch = nn.Conv3d(group, group, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=12, padding=(0, 1, 1),
                               bias=False)
        self.corr_feature_att_8 = ChannelAtt(group, fea_c_up[2])  # group 12
        # self.att_hourglass_8 = Att_Hourglass(group, fea_c_up[3:])
        self.att_hourglass_8 = Att_HourglassFit(group, fea_c_up[3:], final_one=True)
        # self.att_compress_last_8 = nn.Conv3d(group, 1, 3, 1, 1, bias=False)
        # Volume Attention Propagation
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(2 * torch.ones(1))
        self.propagation = Propagation()
        self.propagation_prob = Propagation_prob()
        # Cat Volume
        if not self.attention_weight_only:
            print(self.attention_weight_only)
            self.feature_compress = nn.Sequential(
                                    BasicConv(fea_c_up[1], cat_group, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(cat_group, cat_group//2, 3, 1, 1, bias=False))
            self.sparse_cat_volume = SparseConcatVolume(None)
            # Cat Volume Aggregation
            self.cat_agg_first = BasicConv(cat_group, cat_group//2, is_3d=True, kernel_size=3, stride=1, padding=1)
            self.cat_fea_att_4 = ChannelAtt(cat_group//2, fea_c_up[1])
            self.cat_hourglass_4 = Att_HourglassFit(cat_group//2, fea_c_up[2:4], final_one=True)

        ####################################################################
        # ************************** Spx about  **************************
        ####################################################################
        self.stem_x2 = nn.Sequential(
            BasicConv(3, spxc[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spxc[0], spxc[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[0]),
            nn.ReLU()
        )

        self.stem_x4 = nn.Sequential(
            BasicConv(spxc[0], spxc[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spxc[1], spxc[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[1]),
            nn.ReLU()
        )

        self.stem_x8 = nn.Sequential(
            BasicConv(spxc[1], spxc[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spxc[1], spxc[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spxc[2]),
            nn.ReLU()  # 原代码使用的是的 ReLU
        )

        self.spx_4x = nn.Sequential(
            BasicConv(fea_c_up[1], fea_c[1], kernel_size=3, stride=1, padding=1),  # bn and leakyrelu
            nn.Conv2d(fea_c[1], fea_c[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(fea_c[1]),
            nn.ReLU()
        )
        self.spx_2x = Conv2x(fea_c[1], spxc[0], True)  # Concat  spxc[0] --> 32
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * spxc[0], 9, kernel_size=4, stride=2, padding=1))

    def forward(self, left_img, right_img):
        b, c, h, w = left_img.shape
        disp_pred = []

        cat = torch.cat((left_img, right_img), dim=0)
        f_x2, f_list = self.feature(cat)

        fl_x2, _ = f_x2.split(dim=0, split_size=b)
        f_list = self.up(f_list)

        fl, fr = [], []
        for v_ in f_list:
            fl_, fr_ = v_.split(dim=0, split_size=b)
            fl.append(fl_)
            fr.append(fr_)

        # spx feature
        spxf_x2 = self.stem_x2(cat)
        spxf_x4 = self.stem_x4(spxf_x2)
        spxf_x8 = self.stem_x8(spxf_x4)

        spxfl_x2, _ = spxf_x2.split(dim=0, split_size=b)
        spxfl_x4, spxfr_x4 = spxf_x4.split(dim=0, split_size=b)
        spxfl_x8, spxfr_x8 = spxf_x8.split(dim=0, split_size=b)

        fl[0] = torch.cat((fl[0], spxfl_x4), 1)
        fr[0] = torch.cat((fr[0], spxfr_x4), 1)
        fl[1] = torch.cat((fl[1], spxfl_x8), 1)
        fr[1] = torch.cat((fr[1], spxfr_x8), 1)

        corr_volume = self.corr_volume_construction(fl[1], fr[1], self.D // 8)
        corr_volume = self.patch(corr_volume)
        att_volume = self.corr_feature_att_8(corr_volume, fl[1])
        att_volume = self.att_hourglass_8(att_volume, fl[2:])
        # att_volume = self.att_compress_last_8(att_volume)
        att_weights = F.interpolate(att_volume, [self.D // 4, h // 4, w // 4],
                                    mode='trilinear')

        pred_att = torch.squeeze(att_weights, 1)
        pred_att_prob = F.softmax(pred_att, dim=1)
        pred_att = disparity_regression(pred_att_prob, self.D // 4)  # init
        # Confidence Estimate
        pred_variance = disparity_variance(pred_att_prob, self.D // 4, pred_att.unsqueeze(1))
        pred_variance = self.beta + self.gamma * pred_variance  # Equation 7 in Fast ACV
        pred_variance = torch.sigmoid(pred_variance)
        # Sampling and $ \sigmoid(C_m(i)) $
        pred_variance_samples = self.propagation(pred_variance)

        # disparity sampling
        disparity_samples = self.propagation(pred_att.unsqueeze(1))  # (B, C, D, H, W)
        right_feature_x4, left_feature_x4 = SpatialTransformer_grid(spxfl_x4, spxfr_x4, disparity_samples)
        # Eq.5:  $$S_m = \langle F_l(i) , F_r(i-D(i)) \rangle $$ :
        disparity_sample_strength = (left_feature_x4 * right_feature_x4).mean(dim=1)
        #  Eq. 8 :  W_m(i) = S_m(i) \times \sigmoid (C_m(i)) and generated the related weight by softmax
        disparity_sample_strength = torch.softmax(disparity_sample_strength * pred_variance_samples, dim=1)

        att_weights = self.propagation_prob(att_weights)
        # att_weight is the correlation volume after propagation
        # Eq.9:
        att_weights = att_weights * disparity_sample_strength.unsqueeze(2)
        att_weights = torch.sum(att_weights, dim=1, keepdim=True)
        att_weights_prob = F.softmax(att_weights, dim=2)

        _, ind = att_weights_prob.sort(2, True)
        ind_k = ind[:, :, :self.coarse_topk]
        ind_k = ind_k.sort(2, False)[0]
        att_topk = torch.gather(att_weights_prob, 2, ind_k)
        disparity_sample_topk = ind_k.squeeze(1).float()

        if not self.attention_weight_only:
            concat_fea_l = self.feature_compress(fl[0])
            concat_fea_r = self.feature_compress(fr[0])
            sparse_cat_volume = self.sparse_cat_volume(concat_fea_l, concat_fea_r, disparity_sample_topk)
            # Eq.13
            volume = att_topk * sparse_cat_volume
            volume = self.cat_agg_first(volume)
            volume = self.cat_fea_att_4(volume, fl[0])
            volume_final = self.cat_hourglass_4(volume, fl[1:3])

        # spx
        xspx = self.spx_4x(fl[0])
        xspx = self.spx_2x(xspx, spxfl_x2)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        # disp_pred
        if self.training or self.attention_weight_only:
            att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
            att_prob = F.softmax(att_prob, dim=1)
            pred_att_x4 = torch.sum(att_prob * disparity_sample_topk, dim=1)
            disp_pred.append(pred_att_x4)
            pred_att_x1 = upfeat(pred_att_x4.unsqueeze(dim=1), spx_pred)
            disp_pred.append(pred_att_x1.squeeze(dim=1)*4)
            if self.attention_weight_only:
                return {
                            "preds_pyramid": disp_pred
                        }

        pred_x4 = regression_topk_sparse(volume_final.squeeze(1), self.topk, disparity_sample_topk)
        disp_pred.append(pred_x4)
        pred_up_x1 = upfeat(pred_x4.unsqueeze(dim=1), spx_pred)
        disp_pred.append(pred_up_x1.squeeze(dim=1)*4)

        return {
                    "preds_pyramid": disp_pred
                }

    # @staticmethod
    def get_name(self):
        return self.name