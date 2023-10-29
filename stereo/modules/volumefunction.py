from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def catvolume(left_feature, right_feature, max_disp):
    b, c, h, w = left_feature.shape
    volume = left_feature.new_zeros([b, 2 * c, max_disp, h, w])
    for i in range(max_disp):
        if i > 0:
            volume[:, :c, i, :, :] = left_feature[:, :, :, :]
            volume[:, c:, i, :, i:] = right_feature[:, :, :, :-i]
        else:
            volume[:, :c, i, :, :] = left_feature
            volume[:, c:, i, :, :] = right_feature
    volume = volume.contiguous()
    return volume


def gwcvolume(left_feature, right_feature, max_disp, groups):
    b, c, h, w = left_feature.shape
    volume = left_feature.new_zeros([b, 2 * c, max_disp, h, w])
    assert c % groups == 0
    cpg = c // groups
    for i in range(max_disp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(left_feature[:, :, :, i:], right_feature[:, :, :, :-i],
                                                           groups, cpg)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(left_feature, right_feature, groups, cpg)

    volume = volume.contiguous()
    return volume


def normed_gwcvolume(left_feature, right_feature, max_disp, groups):
    b, c, h, w = left_feature.shape
    volume = left_feature.new_zeros([b, 2 * c, max_disp, h, w])
    assert c % groups == 0
    cpg = c // groups
    for i in range(max_disp):
        if i > 0:
            volume[:, :, i, :, i:] = normed_groupwise_correlation(left_feature[:, :, :, i:], right_feature[:, :, :, :-i],
                                                           groups, cpg)
        else:
            volume[:, :, i, :, :] = normed_groupwise_correlation(left_feature, right_feature, groups, cpg)
    volume = volume.contiguous()
    return volume


def normed_corrvolume(left_feature, right_feature, max_disp):
    b, _, h, w = left_feature.shape
    volume = left_feature.new_zeros([b, 1, max_disp, h, w])
    for i in range(max_disp):
        if i > 0:
            volume[:, :, i, :, i:] = normed_correlation(left_feature[:, :, :, i:], right_feature[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = normed_correlation(left_feature, right_feature)
    volume = volume.contiguous()
    return volume


def corrvolume(left_feature, right_feature, max_disp):
    b, _, h, w = left_feature.shape
    volume = left_feature.new_zeros([b, 1, max_disp, h, w])
    for i in range(max_disp):
        if i > 0:
            volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                   right_feature[:, :, :, :-i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, groups, cpg):
    b, _, h, w = fea1.shape
    cost = (fea1 * fea2).view([b, groups, cpg, h, w]).mean(dim=2)
    assert cost.shape == (b, groups, h, w)
    return cost


def normed_groupwise_correlation(fea1, fea2, groups, cpg):
    b, _, h, w = fea1.shape
    fea1 = fea1.view([b, groups, cpg, h, w])
    fea2 = fea2.view([b, groups, cpg, h, w])
    cost = ((fea1/(torch.norm(fea1, 2, 2, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 2, True)+1e-05))).mean(dim=2)
    assert cost.shape == (b, groups, h, w)
    return cost


def normed_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost