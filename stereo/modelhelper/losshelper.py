import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.logger import Logger as Log
from utils.args_dictionary import get_key


def robust_loss(x, a, c):
    # A general and adaptive robust loss function
    abs_a_sub_2 = abs(a - 2)

    x = x / c
    x = x * x / abs_a_sub_2 + 1
    x = x ** (a / 2)
    x = x - 1
    x = x * abs_a_sub_2 / a
    return x


def loss_helper(args):
    loss_cfg = get_key(args, 'loss')
    max_disp = get_key(args, 'dataset', 'max_disparity')
    if loss_cfg['type'] == "basicdisp":
        loss = DispSmooth(loss_cfg, max_disp, weight=loss_cfg['weight'])
    elif loss_cfg['type'] == "hypdisp":
        loss = HypLoss(loss_cfg, max_disp, weight=None)
    else:
        Log.error("no that loss")
    return loss


class DispSmooth(nn.Module):
    def __init__(self, cfg, max_disp, weight=None):
        super(DispSmooth, self).__init__()

        self.loss_weight = weight
        self.max_disp = max_disp
        self.highest_only = cfg.get("highest_only", False)
        if self.highest_only:
            self.loss_weight = [1]
        self.pseudo_gt = cfg.get("pseudo_gt", False)

    def forward(self, preds_dict, target, pseudo_disp=None, **kwargs):
        # get the disp
        for name in preds_dict.keys():
            if ("preds" in name) and ("pyramid" in name):
                preds = preds_dict[name]

        if self.pseudo_gt and (pseudo_disp is None):
            Log.error('Pseudo ground truth could not be None when adopted')
            exit(1)

        mask = (target > 1e-3) & (target < self.max_disp)

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

        # return total_loss, disp_loss, pyramid_loss, pseudo_disp_loss, pseudo_pyramid_loss
        return {
            "total_loss": total_loss,
            "multi_preds_loss": disp_loss,
            "multi_preds_pyramid": pyramid_loss,
            "pseudo_preds_loss": pseudo_disp_loss,
            "pseudo_preds_pyramid": pseudo_pyramid_loss
        }

    def loss_tb_logger(self, tb_logger, loss_dic, i_iter):

        if "multi_preds_pyramid" in loss_dic.keys():
            pyramid_loss = loss_dic["multi_preds_pyramid"]
            for s in range(len(pyramid_loss)):
                save_name = 'train/loss' + str(len(pyramid_loss) - s - 1)
                save_value = pyramid_loss[s]
                tb_logger.add_scalar(save_name, save_value, i_iter)

        if self.pseudo_gt:

            if "pseudo_preds_loss" in loss_dic.keys():
                pseudo_preds_loss = loss_dic["pseudo_preds_loss"]
                tb_logger.add_scalar('train/pseudo_loss', pseudo_preds_loss.item(), i_iter)

            if "pseudo_preds_pyramid" in loss_dic.keys():
                pyramid_loss = loss_dic["pseudo_preds_pyramid"]
                for s in range(len(pyramid_loss)):
                    save_name = 'train/pseudo_loss' + str(len(pyramid_loss) - s - 1)
                    save_value = pyramid_loss[s]
                    tb_logger.add_scalar(save_name, save_value, i_iter)


class HypLoss(nn.Module):
    def __init__(self, cfg, max_disp, weight=None):
        super(HypLoss, self).__init__()
        self.slant = True
        self.max_disp = max_disp
        self.loss_weight = weight
        # hyper-parameter
        parameters = cfg.get('parameters')
        self.A = parameters['A']
        self.B = parameters['B']
        self.C1 = parameters['C1']
        self.C2 = parameters['C2']
        self.robust_a = parameters['robust_a']
        self.robust_c = parameters['robust_c']
        self.beta = parameters['beta']  # Eq 10 margin
        self.k = parameters["init_k"]

        self.last_prop = 3

        self.highest_only = cfg.get("highest_only", False)
        if self.highest_only:
            self.loss_weight = [1]
        self.pseudo_gt = cfg.get("pseudo_gt", False)

    def multi_scale_loss(self, diff, mask, A=None):
        # Eq. 14
        if A is not None:
            closer_mask = diff < self.A
        elif A is None:
            # For the last several levels, when only a single hypotheses is
            # available, loss is applied to all pixels
            closer_mask = mask
        mask = mask * closer_mask

        loss = robust_loss(diff, a=self.robust_a, c=self.robust_c)
        return (loss * mask).sum() / (mask.sum() + 1e-6)

    def init_loss(self, volume, target, tile_size=4):
        # compute the matching cost for subpixel disparities using linear interpolation
        target = target.unsqueeze(1)
        scale = target.size(-1) // volume.size(-1)
        scale_disp = max(1, scale // tile_size)

        target = target / scale_disp
        max_disp = self.max_disp / scale_disp

        target = F.max_pool2d(target, kernel_size=scale, stride=scale)
        # To compute them at multiple resolutions we maxpool the ground
        # truth disparity maps to downsample them
        mask = (target < max_disp) & (target > 1e-3)

        def rho(d):  # ρ(d)
            d = torch.clip(d, 0, volume.size(1) - 1)
            return torch.gather(volume, dim=1, index=d)

        def phi(d):  # φ(d)
            df = torch.floor(d).long()
            d_sub_df = d - df
            # linear interpolation
            return d_sub_df * rho(df + 1) + (1 - d_sub_df) * rho(df)

        pixels = mask.sum() + 1e-6
        gt_loss = (phi(target) * mask).sum() / pixels

        d_range = torch.arange(0, max_disp, dtype=target.dtype, device=target.device)
        d_range = d_range.view(1, -1, 1, 1)
        d_range = d_range.repeat(target.size(0), 1, target.size(-2), target.size(-1))

        low = target - 1.5
        up = target + 1.5
        d_range_mask = (low <= d_range) & (d_range <= up) | (~mask)

        cv_nm = torch.masked_fill(volume, d_range_mask, float("inf"))
        cost_nm = torch.topk(cv_nm, k=self.k, dim=1, largest=False).values

        nm_loss = torch.clip(self.beta - cost_nm, min=0)
        nm_loss = (nm_loss * mask).sum() / pixels
        return gt_loss + nm_loss

    def slant_loss(self, dx, dy, dx_gt, dy_gt, d_diff, mask):
        # Eq.13
        if len(dx.size()) == 4:
            dx = dx.squeeze(dim=1)
            dy = dy.squeeze(dim=1)
        closer_mask = d_diff < self.B
        mask = mask * closer_mask  # mask and
        slant_diff = torch.cat([(dx_gt - dx), (dy_gt - dy)], dim=1)
        loss = torch.norm(slant_diff, p=1, dim=1, keepdim=False)
        loss = (loss * mask).sum() / (mask.sum()+1e-6)
        # print('slant_loss: {:.3f}'.format(loss.mean()))
        # print('dx_gt mean: {:.3f}, dy_gt mean: {:.3f}'.format(dx_gt.mean(), dy_gt.mean()))
        return loss  # 1-dim vector

    def confidence(self, conf, diff, mask):
        closer_mask = diff < self.C1
        further_mask = diff > self.C2
        mask = mask * (closer_mask + further_mask)  # mask and
        closer_item = F.relu(1 - conf)
        further_item = F.relu(conf)
        # pdb.set_trace()
        loss = closer_item * closer_mask.float() + further_item * further_mask.float()
        loss = (loss * mask).sum() / (mask.sum()+1e-6)
        return loss  # 1-dim vector

    def forward(self, preds_dict, target, dxygt=None, pseudo_disp=None, **kwargs):
        """
                "preds_pyramid": preds_pyramid,
                "preds_pyramid_coarse": preds_pyramid_coarse,
                "slant_pyramid": slant_pyramid,
                "slant_pyramid_coarse": slant_pyramid_coarse,
                "confidence_pyramid": conf_pyramid,
                "confidence_pyramid_coarse": conf_pyramid_coarse,
                "volume_pyramid": volume_pyramid,
        """
        # get the different input
        for name in preds_dict.keys():
            if ("preds" in name) and ("pyramid" in name):
                if "coarse" not in name:
                    preds = preds_dict[name]
                elif "coarse" in name:
                    preds_coarse = preds_dict[name]
            elif ("slant" in name) and ("pyramid" in name):
                if "coarse" not in name:
                    slant_set = preds_dict[name]
                elif "coarse" in name:
                    slant_coarse = preds_dict[name]
            elif ("confidence" in name) and ("pyramid" in name):
                if "coarse" not in name:
                    conf_set = preds_dict[name]
                elif "coarse" in name:
                    conf_coarse = preds_dict[name]
            elif ("volume" in name) and ("pyramid" in name):
                volume_set = preds_dict[name]

        if self.pseudo_gt and (pseudo_disp is None):
            Log.error('Pseudo ground truth could not be None when adopted')
            exit(1)

        if self.slant and (dxygt is None):
            Log.error('Slant truth could not be None!!')
            exit(1)

        mask = (target > 0) & (target < self.max_disp)

        if self.pseudo_gt:
            pseudo_mask = (pseudo_disp > 0) & (pseudo_disp < self.max_disp) & (~mask)  # inverse mask

        if not isinstance(preds, list):
            preds = [preds]

        if self.highest_only:
            preds = [preds[-1]]  # only the last highest resolution output

        scale_loss, init_loss, slant_loss, conf_loss, pseudo_disp_loss = 0, 0, 0, 0, 0
        scale_pyramid, init_pyramid, slant_pyramid, conf_pyramid, pseudo_pyramid_loss, diff_set = [], [], [], [], [], []
        scale_pyramid_coarse, slant_pyramid_coarse, conf_pyramid_coarse, diff_set_coarse = [], [], [], []

        mask = (target < self.max_disp) & (target > 1e-3)

        for i in range(len(preds)):
            diff = abs(preds[i]-target)
            diff_set.append(diff)
            if i >= len(preds)-self.last_prop:
                current_scale = self.multi_scale_loss(diff, mask)
            elif i < len(preds)-self.last_prop:
                current_scale = self.multi_scale_loss(diff, mask, self.A)
            scale_pyramid.append(current_scale)
            scale_loss = scale_loss + current_scale
        for i in range(len(preds_coarse)):
            diff = abs(preds_coarse[i]-target)
            diff_set_coarse.append(diff)
            current_scale = self.multi_scale_loss(diff, mask)
            scale_pyramid_coarse.append(current_scale)
            scale_loss = scale_loss + current_scale

        if volume_set is not None:
            for i in range(len(volume_set)):
                current_init = self.init_loss(volume_set[i], target, tile_size=4)
                init_pyramid.append(current_init)
                init_loss = init_loss + current_init

        if slant_set is not None:
            for i in range(len(slant_set)):
                current_slant = self.slant_loss(slant_set[i][:, :1], slant_set[i][:, 1:], dxygt[:, :1], dxygt[:, 1:], diff_set[i], mask)
                slant_pyramid.append(current_slant)
                slant_loss = slant_loss + current_slant
        if slant_coarse is not None:
            for i in range(len(slant_coarse)):
                current_slant = self.slant_loss(slant_coarse[i][:, :1], slant_coarse[i][:, 1:], dxygt[:, :1], dxygt[:, 1:], diff_set_coarse[i], mask)
                slant_pyramid_coarse.append(current_slant)
                slant_loss = slant_loss + current_slant

        if conf_set is not None:
            for i in range(len(conf_set)):
                w = conf_set[i]
                current_confidence = self.confidence(w, diff_set[i+1], mask)  # no confidence in the lowest disp map
                conf_loss = conf_loss + current_confidence
                conf_pyramid.append(current_confidence)
        if conf_coarse is not None:
            for i in range(len(conf_coarse)):
                w = conf_coarse[i]
                current_confidence = self.confidence(w, diff_set_coarse[i], mask)
                conf_loss = conf_loss + current_confidence
                conf_pyramid_coarse.append(current_confidence)

        total_loss = scale_loss + init_loss + slant_loss + conf_loss

        # return total_loss, scale_loss, init_loss, slant_loss, conf_loss

        return {
            "total_loss": total_loss,
            "multi_preds_loss": scale_loss,
            "multi_preds_pyramid": scale_pyramid,
            "multi_preds_pyramid_coarse": scale_pyramid_coarse,
            "init_loss": init_loss,
            "init_pyramid": init_pyramid,
            "slant_loss": slant_loss,
            "slant_pyramid": slant_pyramid,
            "slant_pyramid_coarse": slant_pyramid_coarse,
            "confidence_loss": conf_loss,
            "confidence_pyramid": conf_pyramid,
            "confidence_pyramid_coarse": conf_pyramid_coarse,
        }

    def loss_tb_logger(self, tb_logger, loss_dic, i_iter):
        for name in loss_dic.keys():
            if "multi_preds_pyramid" in name and ("coarse" not in name):
                pyramid_loss = loss_dic[name]
                tag = "disp_loss"
            elif "multi_preds_pyramid" in name and ("coarse" in name):
                pyramid_loss = loss_dic[name]
                tag = "disp_coarse_loss"
            elif "init_pyramid" in name:
                pyramid_loss = loss_dic[name]
                tag = "init_loss"
            elif "slant_pyramid" in name and ("coarse" not in name):
                pyramid_loss = loss_dic[name]
                tag = "slant_loss"
            elif "slant_pyramid" in name and ("coarse" in name):
                pyramid_loss = loss_dic[name]
                tag = "slant_coarse_loss"
            elif "confidence_pyramid" in name and ("coarse" not in name):
                pyramid_loss = loss_dic[name]
                tag = "conf_loss"
            elif "confidence_pyramid" in name and ("coarse" in name):
                pyramid_loss = loss_dic[name]
                tag = "conf_coarse_loss"
            else:
                continue
            for s in range(len(pyramid_loss)):
                save_name = 'train/' + tag + str(len(pyramid_loss) - s - 1)
                save_value = pyramid_loss[s]
                tb_logger.add_scalar(save_name, save_value, i_iter)

        if "init_loss" in loss_dic.keys():
            init_loss = loss_dic["init_loss"]
            tb_logger.add_scalar('train/init_loss', init_loss.item(), i_iter)

        if "slant_loss" in loss_dic.keys():
            slant_loss = loss_dic["slant_loss"]
            tb_logger.add_scalar('train/slant_loss', slant_loss.item(), i_iter)

        if "confidence_loss" in loss_dic.keys():
            confidence_loss = loss_dic["confidence_loss"]
            tb_logger.add_scalar('train/confidence_loss', confidence_loss.item(), i_iter)









