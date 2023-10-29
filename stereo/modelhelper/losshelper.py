import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.logger import Logger as Log
from utils.args_dictionary import get_key


def loss_helper(args):
    loss_cfg = get_key(args, 'loss')
    max_disp = get_key(args, 'dataset', 'max_disparity')
    if loss_cfg['type'] == "basicdisp":
        loss = DispSmooth(args, max_disp, weight=loss_cfg['weight'])

    return loss


class DispSmooth(nn.Module):
    def __init__(self, args, max_disp, weight=None, highest_only=False, pseudo_gt=False):
        super(DispSmooth, self).__init__()
        self.loss_weight = weight
        self.max_disp = max_disp
        self.highest_only = highest_only
        if highest_only:
            self.loss_weight = [1]
        self.pseudo_gt = pseudo_gt

    def forward(self, preds, target, mask, pseudo_disp=None):

        if self.pseudo_gt and (pseudo_disp is None):
            Log.error('Pseudo ground truth could not be None when adopted')
            exit(1)

        # mask = (target > 0) & (target < self.max_disp)

        if self.pseudo_gt:
            pseudo_mask = (pseudo_disp > 0) & (pseudo_disp < self.max_disp) & (~mask)  # inverse mask

        if not isinstance(preds, list):
            preds = [preds]

        if self.highest_only:
            preds = [preds[-1]]  # only the last highest resolution output

        if len(preds) != len(self.loss_weight):
            Log.error('Len of the loss weight does match the preds')
            exit(1)

        disp_loss = 0
        pseudo_disp_loss = 0
        pyramid_loss = []
        pseudo_pyramid_loss = []

        for k in range(len(preds)):
            pred_disp = preds[k]
            weight = self.loss_weight[k]

            if pred_disp.size(-1) != target.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = F.interpolate(pred_disp, size=(target.size(-2), target.size(-1)),
                                          mode='bilinear', align_corners=False) * (target.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]

            curr_loss = F.smooth_l1_loss(pred_disp[mask], target[mask], reduction='mean')
            disp_loss += weight * curr_loss
            pyramid_loss.append(curr_loss)

            if self.pseudo_gt:
                pseudo_curr_loss = F.smooth_l1_loss(pred_disp[pseudo_mask], pseudo_disp[pseudo_mask],
                                                    reduction='mean')
                pseudo_disp_loss += weight * pseudo_curr_loss
                pseudo_pyramid_loss.append(pseudo_curr_loss)

        total_loss = disp_loss + pseudo_disp_loss

        return total_loss, disp_loss, pyramid_loss, pseudo_disp_loss, pseudo_pyramid_loss






