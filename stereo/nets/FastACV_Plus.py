from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.logger import Logger as Log
from utils.args_dictionary import get_key

from stereo.modules.submodule import BasicConv, Conv2x, ChannelAtt
from stereo.modules.feature import BasicFeature, BasicFeaUp
from stereo.modules.volumehelper import VolumeHelper, SparseConcatVolume
from stereo.modules.aggregation import Att_HourglassFit
from stereo.modules.regression import regression_topk_sparse, disparity_regression
from stereo.modules.spixel import upfeat


class FastACV_Plus_Original(nn.Module):
    def __init__(self, args):
        super(FastACV_Plus_Original, self).__init__()

        self.name = "FastACV_Plus_Original"
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

        ####################################################################
        # ********* Corr Volume and Concatenation Volume Structure  ********
        ####################################################################
        # ACV
        self.coarse_topk = acv_cfg["coarse_topk"]
        self.attention_weight_only = acv_cfg["att_weights_only"]  # top k value in the coarse map

        att_volume_set = get_key(args, 'net', "volume1")
        cat_volume_set = get_key(args, 'net', "volume2")
        self.corr_volume_construction = VolumeHelper(att_volume_set)
        group = att_volume_set["group"]  # in fact group = 1
        cat_group = cat_volume_set["group"]  # 32
        # Corr Volume aggregation  fea_c_up[1] --> 96
        self.corr_fea_compress_x4 = nn.Sequential(
            BasicConv(fea_c_up[1], fea_c_up[1]//2, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(fea_c_up[1]//2, fea_c_up[1]//2, kernel_size=1, padding=0, stride=1))
        self.corr_agg_4x_first = BasicConv(group, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att_4x = ChannelAtt(8, fea_c_up[1])
        self.att_hourglass_4 = Att_HourglassFit(8, fea_c_up[2:], final_one=True)  # Att_HourglassFit(8, fea_c_up[1:], final_one=True)

        # Cat Volume
        if not self.attention_weight_only:
            self.feature_compress = nn.Sequential(
                                    BasicConv(fea_c_up[1], cat_group, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(cat_group, cat_group//2, 3, 1, 1, bias=False))
            self.sparse_cat_volume = SparseConcatVolume(None)
            # Cat Volume Aggregation
            self.cat_agg_first = BasicConv(cat_group, cat_group // 2, is_3d=True, kernel_size=3, stride=1, padding=1)
            self.cat_fea_att_4 = ChannelAtt(cat_group // 2, fea_c_up[1])

            self.cat_hourglass_4 = Att_HourglassFit(cat_group // 2, fea_c_up[2:4], final_one=True)  # Att_Hourglass(cat_group // 2, fea_c_up[2:4])

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
        spxfl_x2, _ = spxf_x2.split(dim=0, split_size=b)
        spxfl_x4, spxfr_x4 = spxf_x4.split(dim=0, split_size=b)

        fl[0] = torch.cat((fl[0], spxfl_x4), 1)
        fr[0] = torch.cat((fr[0], spxfr_x4), 1)

        # generate the corr volume
        match_feal = self.corr_fea_compress_x4(fl[0])
        match_fear = self.corr_fea_compress_x4(fr[0])
        corr_volume = self.corr_volume_construction(match_feal, match_fear, self.D // 4)
        corr_volume = self.corr_agg_4x_first(corr_volume)
        att_volume = self.corr_feature_att_4x(corr_volume, fl[0])
        att_weights = self.att_hourglass_4(att_volume, fl[1:])
        att_weights_prob = F.softmax(att_weights, dim=2)

        # coarse map top k
        _, ind = att_weights_prob.sort(2, True)
        ind_k = ind[:, :, :self.coarse_topk]
        ind_k = ind_k.sort(2, False)[0]
        att_topk = torch.gather(att_weights_prob, 2, ind_k)
        disparity_sample_topk = ind_k.squeeze(1).float()

        if not self.attention_weight_only:
            concat_fea_l = self.feature_compress(fl[0])
            concat_fea_r = self.feature_compress(fr[0])
            sparse_cat_volume = self.sparse_cat_volume(concat_fea_l, concat_fea_r, disparity_sample_topk)
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
            disp_pred.append(pred_att_x1.squeeze(dim=1) * 4)
            if self.attention_weight_only:
                return {
                        "preds_pyramid": disp_pred
                       }

        pred_x4 = regression_topk_sparse(volume_final.squeeze(1), self.topk, disparity_sample_topk)
        disp_pred.append(pred_x4)
        pred_up_x1 = upfeat(pred_x4.unsqueeze(dim=1), spx_pred)
        disp_pred.append(pred_up_x1.squeeze(dim=1) * 4)
        return {
                    "preds_pyramid": disp_pred
                }

    # @staticmethod
    def get_name(self):
        return self.name