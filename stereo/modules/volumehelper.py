import torch
import torch.nn as nn
from utils.logger import Logger as Log

def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


class ConcatVolume(nn.Module):
    def __init__(self, cfg):
        super(ConcatVolume, self).__init__()
        """
                # AANet
                cost_volume = left_feature.new_zeros(b, 2 * c, maxdisp, h, w)
                    for i in range(maxdisp):
                        if i > 0:
                            cost_volume[:, :, i, :, i:] = torch.cat((left_feature[:, :, :, i:], right_feature[:, :, :, :-i]),
                                                                    dim=1)
                        else:
                            cost_volume[:, :, i, :, :] = torch.cat((left_feature, right_feature), dim=1)
         """

    def forward(self, left_feature, right_feature, max_disp = None):
        B, C, H, W = left_feature.shape
        volume = left_feature.new_zeros([B, 2 * C, max_disp, H, W])
        # FastACV
        for i in range(max_disp):
            if i > 0:
                volume[:, :C, i, :, :] = left_feature[:, :, :, :]
                volume[:, C:, i, :, i:] = right_feature[:, :, :, :-i]
            else:
                volume[:, :C, i, :, :] = left_feature
                volume[:, C:, i, :, :] = right_feature
        volume = volume.contiguous()
        return volume

class CorVolume(nn.Module):
    """
    Output of CorVolume is 3D
    to obtain 4D CorVolume, use GwcVolume and set group as 1
    """
    def __init__(self, cfg):
        super(CorVolume, self).__init__()
        self.norm = cfg['norm'] if 'norm' in cfg else False  # group

    def _norm_correlation(self, fea1, fea2):
        cvolume = torch.mean(
            ((fea1 / (torch.norm(fea1, 2, 1, True) + 1e-05)) * (fea2 / (torch.norm(fea2, 2, 1, True) + 1e-05))), dim=1,
            keepdim=True)
        return cvolume

    def forward(self, left_feature, right_feature, max_disp=None):
        b, c, h, w = left_feature.size()
        volume = left_feature.new_zeros(b, max_disp, h, w)
        if not self.norm:
            for i in range(max_disp):
                if i > 0:
                    volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                            right_feature[:, :, :, :-i]).mean(dim=1)
                else:
                    volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
        elif self.norm:
            for i in range(max_disp):
                if i > 0:
                    volume[:, :, i, :, i:] = self._norm_correlation(left_feature[:, :, :, i:], right_feature[:, :, :, :-i])
                else:
                    volume[:, :, i, :, :] = self._norm_correlation(left_feature, right_feature)
        volume = volume.contiguous()
        return volume


class GwcVolume(nn.Module):
    def __init__(self, cfg):
        super(GwcVolume, self).__init__()
        self.group = cfg['group']  # group
        self.norm = cfg['norm'] if 'norm' in cfg else False  # group

    """
    # CoEx+Glue
                class CostVolume(nn.Module):
            def __init__(self, maxdisp, glue=False, group=1):
                super(CostVolume, self).__init__()
                self.maxdisp = maxdisp+1
                self.glue = glue
                self.group = group
                self.unfold = nn.Unfold((1, maxdisp+1), 1, 0, 1)
                self.left_pad = nn.ZeroPad2d((maxdisp, 0, 0, 0))

            def forward(self, x, y, v=None):
                b, c, h, w = x.shape

                unfolded_y = self.unfold(self.left_pad(y)).reshape(
                    b, self.group, c//self.group, self.maxdisp, h, w)
                x = x.reshape(b, self.group, c//self.group, 1, h, w)

                cost = (x*unfolded_y).sum(2)
                cost = torch.flip(cost, [2])

                if self.glue:
                    cross = self.unfold(self.left_pad(v)).reshape(
                        b, c, self.maxdisp, h, w)
                    cross = torch.flip(cross, [2])
                    return cost, cross
                else:
                    return cost
    """

    def _groupwise_correlation(self, fea1, fea2, cpg):
        # gwc_paras = [B, C, H, W, self.group, channels_per_group]
        b, c, h, w = fea1.shape
        if not self.norm:
            cost = (fea1 * fea2).view([b, self.group, cpg, h, w]).mean(dim=2)
        elif self.norm:
            fea1 = fea1.view([b, self.group, cpg, h, w])
            fea2 = fea2.view([b, self.group, cpg, h, w])
            cost = ((fea1 / (torch.norm(fea1, 2, 2, True) + 1e-05)) * (
                        fea2 / (torch.norm(fea2, 2, 2, True) + 1e-05))).mean(dim=2)
        assert cost.shape == (b, self.group, h, w)  # (B, num_groups, H, W)
        return cost

    def forward(self, left_feature, right_feature, max_disp=None):
        b, c, h, w = left_feature.shape
        assert c % self.group == 0
        cpg = c // self.group
        volume = left_feature.new_zeros([b, self.group, max_disp, h, w])
        for i in range(max_disp):
            if i > 0:
                volume[:, :, i, :, i:] = self._groupwise_correlation(left_feature[:, :, :, i:], right_feature[:, :, :, :-i],
                                                               cpg)
            else:
                volume[:, :, i, :, :] = self._groupwise_correlation(left_feature, right_feature, cpg)
        volume = volume.contiguous()
        return volume


class DiffVolume(nn.Module):
    def __init__(self, cfg):
        super(DiffVolume, self).__init__()
        # self.group = cfg.get('group')  # group

    def forward(self, left_feature, right_feature, max_disp=None):
        b, c, h, w = left_feature.size()
        volume = left_feature.new_zeros(b, c, max_disp, h, w)

        for i in range(self.max_disp):
            if i > 0:
                volume[:, :, i, :, i:] = left_feature[:, :, :, i:] - right_feature[:, :, :, :-i]
            else:
                volume[:, :, i, :, :] = left_feature - right_feature

        volume = volume.contiguous()
        return volume


class VolumeHelper(nn.Module):
    def __init__(self, cfg, feature_similarity = None, group = None):
        """Construct cost volume based on different
                similarity measures
            Args: cfg --> configer
        """
        super(VolumeHelper, self).__init__()
        feature_similarity = cfg['feature_similarity']

        if feature_similarity == "concat":
            self.volumeconstructor = ConcatVolume(cfg)
        elif feature_similarity == "groupcorrelation":
            self.volumeconstructor = GwcVolume(cfg)
        elif feature_similarity == "correlation":
            self.volumeconstructor = CorVolume(cfg)
        elif feature_similarity == "difference":
            self.volumeconstructor = DiffVolume(cfg)
        else:
            Log.error('Volume Type: {} not valid!'.format(feature_similarity))
            exit(1)
        Log.info('Use Volume Type: {}.'.format(feature_similarity))

    def forward(self, left_feature, right_feature, max_disp):
        volume = self.volumeconstructor(left_feature, right_feature, max_disp)
        return volume


