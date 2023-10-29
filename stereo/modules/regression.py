#!/usr/bin/env python
import torch
import torch.nn.functional as F


def disparity_regression(x, maxdisp):
    # print(x.shape)
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def regression_topk(volume, k, maxdisp):
    if k == 1:
        _, ind = volume.sort(1, True)
        pool_ind_ = ind[:, :k]
        b, _, h, w = pool_ind_.shape
        pool_ind = pool_ind_.new_zeros((b, 3, h, w))
        pool_ind[:, 1:2] = pool_ind_
        pool_ind[:, 0:1] = torch.max(
            pool_ind_ - 1, pool_ind_.new_zeros(pool_ind_.shape))
        pool_ind[:, 2:] = torch.min(
            pool_ind_ + 1, maxdisp * pool_ind_.new_ones(pool_ind_.shape))
        corr = torch.gather(volume, 1, pool_ind)

        disp = pool_ind

    else:
        _, ind = volume.sort(1, True)
        pool_ind = ind[:, :k]
        corr = torch.gather(volume, 1, pool_ind)
        disp = pool_ind

    corr = F.softmax(corr, 1)
    disp_4 = torch.sum(corr * disp, 1, keepdim=False)

    return disp_4