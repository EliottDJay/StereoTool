import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
import math


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()


######################################################################
# ******************* Basic utilization or in CoEx  ********************
# Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation
# ####################################################################

class ChannelAtt(SubModule):
    def __init__(self, cv_chan, im_chan, D=None):
        super(ChannelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.weight_init()

    def forward(self, volume, im):
        '''
        im : B C H W
        volume : B G C H W --> in fact  G --> cv_chan
        att : B cv_chan H W  --> after unsqueeze B cv_chan 1 H W broadcast to volume
        '''
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att)*volume
        return cv


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                self.bn = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = self.LeakyReLU(x)#, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


######################################################################
# ********************* utilization in ACVFast  **********************
# Accurate and Efﬁcient Stereo Matching via Attention Concatenation Volume
# ####################################################################

def disparity_variance(x, maxdisp, disparity):
    """
    the shape of disparity should be B,1,H,W, return is the variance of the cost volume [B,1,H,W]
    Equation 6 in the original paper Fast ACV
    $$ U(i) = \sum_{d=0}^{D_{}max}} (P) \times (d - D^{init})^2 $$
    :param x: Confidence Volume with probability in every disparity search space
    :param maxdisp:
    :param disparity: generated coarse disparity map with disparity regression to x
    :return:
    """
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    disp_values = (disp_values - disparity) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)


class Propagation(nn.Module):
    """
    BY Original Fast ACV
    """
    def __init__(self):
        super(Propagation, self).__init__()
        self.replicationpad = nn.ReplicationPad2d(1)

    def forward(self, disparity_samples):
        one_hot_filter = torch.zeros(5, 1, 3, 3, device=disparity_samples.device).float()
        one_hot_filter[0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 2] = 1.0
        disparity_samples = self.replicationpad(disparity_samples)
        aggregated_disparity_samples = F.conv2d(disparity_samples,
                                                one_hot_filter, padding=0)

        return aggregated_disparity_samples


class Propagation_prob(nn.Module):
    def __init__(self):
        super(Propagation_prob, self).__init__()
        self.replicationpad = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))

    def forward(self, prob_volume):
        one_hot_filter = torch.zeros(5, 1, 1, 3, 3, device=prob_volume.device).float()
        one_hot_filter[0, 0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 0, 2] = 1.0

        prob_volume = self.replicationpad(prob_volume)
        prob_volume_propa = F.conv3d(prob_volume, one_hot_filter,padding=0)


        return prob_volume_propa


def SpatialTransformer_grid(x, y, disp_range_samples):
    """
    BY Original Fast ACV
    Though the coarse disparity map, the function generate the related warp feature map
    :param x: left img feature
    :param y: right img feature
    :param disp_range_samples: generated coarse disparity map usually after propagation sampled
    :return:
    """

    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size()[1]

    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)

    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, H, W, 2)

    y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                               padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height, width)  #(B, C, D, H, W)

    x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1)  # (B, C, D, H, W)

    return y_warped, x_warped


######################################################################
# ********************* utilization in HITNet  **********************
# HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching
# https://github.com/zjjMaiMai/TinyHITNet
# ####################################################################


@functools.lru_cache()
@torch.no_grad()
def warp_coef(scale, device):
    """
    if scale=2, return [-0.5, 0.5]
    if scale=4, related to the tile mode
    :param scale: mainly 2 or 4
    :param device: img.device
    :return:
    """
    center = (scale - 1) / 2
    index = torch.arange(scale, device=device) - center
    coef_y, coef_x = torch.meshgrid(index, index, indexing="ij")
    coef_x = coef_x.reshape(1, -1, 1, 1)
    coef_y = coef_y.reshape(1, -1, 1, 1)
    return coef_x, coef_y


def disp_up(d, dx, dy, scale, tile):
    """

    :param d:
    :param dx:
    :param dy:
    :param scale:
    :param tile:
    :return: shape --> [b, 1, h * scale, w  * scale]
    """
    b, _, h, w = d.size()
    coef_x, coef_y = warp_coef(scale, d.device)

    if tile:
        # Eq 5:
        d = d + coef_x * dx + coef_y * dy
    else:
        # use in propagation. coarse hypothesis should 2x first and coef there are 1/2, so scale coefficient is set to 4
        # 不知道怎么说 好像是这样的
        d = d * scale + coef_x * dx * 4 + coef_y * dy * 4

    """d = d.reshape(b, 1, scale, scale, h, w)
    d = d.permute(0, 1, 4, 2, 5, 3)
    d = d.reshape(b, 1, h * scale, w * scale)"""

    d = F.pixel_shuffle(d, scale)

    return d


def hyp_up(hyp, scale=1, tile_scale=1):
    if scale != 1:
        d = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile=False)  # scale = 2
        p = F.interpolate(hyp[:, 1:], scale_factor=scale)
        hyp = torch.cat((d, p), dim=1)
    if tile_scale != 1:
        d = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], tile_scale, tile=True)
        p = F.interpolate(hyp[:, 1:], scale_factor=tile_scale)
        hyp = torch.cat((d, p), dim=1)
    return hyp


class TileExpand(nn.Module):
    def __init__(self, upscale, tile_size=4):
        super(TileExpand, self).__init__()
        self.upscale = upscale
        self.center = (upscale - 1) / 2
        self.DUC = nn.PixelShuffle(upscale)
        self.tile_size = tile_size
        self.dispupsize = self.upscale / self.tile_size

    def forward(self, disp, dx, dy, squeeze_dim=None):
        # b, _, h, w = disp.size()
        disp_upsample = disp * self.dispupsize
        coef_x, coef_y = warp_coef(self.upscale, disp.device)
        disp_upsample = disp_upsample + coef_x * dx + coef_y * dy

        # disp_upsample = F.pixel_shuffle(disp_upsample, self.upscale)
        disp_upsample = self.DUC(disp_upsample)

        if squeeze_dim:
            disp_upsample = torch.squeeze(disp_upsample, dim=squeeze_dim)

        return disp_upsample


class SlantKeepUpsample(nn.Module):
    """
    Slant map upsampling, with tile size keeping
    Here, we use it when upsample coarse hyp with 4x4 tile size
    and the generated upsampled target having the same tile size
    Example:
    The original 4x4 tile used in HITNet usually expand with the mesh [-1.5, -0.5, 0.5, -1.5] x [-1.5, -0.5, 0.5, -1.5]
    refer to the above function TileExpand
    Therefore, to keep the same tile size when upsampling,
    a larger sampling mesh should be used to keep room for implicit tile
    -2 -1.5 -1 -0.5 0 0.5 1 1.5 2
     x   -      -   |  -     -  x
    """
    def __init__(self, upscale, tile_size=4):
        super(SlantKeepUpsample, self).__init__()
        self.upscale = upscale
        self.tile_size = tile_size
        self.DUC = nn.PixelShuffle(upscale)

    def forward(self, disp, dx, dy):
        disp_upsample = disp * self.upscale
        coef_x, coef_y = warp_coef(self.upscale, disp.device)

        disp_upsample = disp_upsample + coef_x * dx * self.tile_size + coef_y * dy * self.tile_size
        # disp_upsample = F.pixel_shuffle(disp_upsample, self.upscale)
        disp_upsample = self.DUC(disp_upsample)

        return disp_upsample


class SlantChangeUpsample(nn.Module):
    """
    Slant upsampling, with tile size changing
    Here, we use it when upsample coarse hyp with 4x4 tile size
    and the generated upsampled target having the tile size 2x2 in the project
    Example:
    The original 4x4 tile used in HITNet usually expand with the mesh [-1.5, -0.5, 0.5, -1.5] x [-1.5, -0.5, 0.5, -1.5]
    To obtain tile with 2x2 when upsampling,
    a shifted mesh should be used for smaller tile
    -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5
         -      -   |  -     -  _  -     -  |  -     -
            x      |     x           x      |     x
        -   |   -    -   |   -
    """
    def __init__(self, upscale, original_size=4, change_size=2):
        super(SlantChangeUpsample, self).__init__()
        assert upscale == original_size/change_size
        self.upscale = upscale
        self.original_tile_size = original_size
        self.changed_tile_size = change_size
        self.DUC = nn.PixelShuffle(upscale)

    def forward(self, disp, dx, dy):
        coef_x, coef_y = warp_coef(self.upscale, disp.device)

        disp_upsample = disp + coef_x * dx * self.upscale + coef_y * dy * self.upscale
        disp_upsample = self.DUC(disp_upsample)

        return disp_upsample


def warp(img, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    img: [B, C, H, W] (im2)
    disp: [B, 1, H, W]
    Tips, dips should clip to -disp, 0 or v should be clipped to 0, max W
    output:
    """
    B, C, H, W = img.size()
    # mesh grid
    x_index = torch.arange(W, device=img.device)
    y_index = torch.arange(H, device=img.device)
    coef_y, coef_x = torch.meshgrid(y_index, x_index, indexing="ij")
    coef_y = coef_y.view(1, 1, H, W).repeat(B, 1, 1, 1)
    coef_x = coef_x.view(1, 1, H, W).repeat(B, 1, 1, 1)

    coef_dx = coef_x - disp
    coef_dx = torch.clip(coef_dx, min=0, max=W-1)

    coef_dx = 2.0 * coef_dx/max(W-1, 1) - 1.0
    coef_y = 2.0 * coef_y/max(H-1, 1) - 1.0

    vgrid = torch.cat((coef_dx, coef_y), dim=1).float()

    """xx = torch.arange(0, W, device=img.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=img.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float()
    # vgrid = Variable(grid)
    vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp
    vgrid = torch.clip(vgrid)
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    """

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(img, vgrid)
    return output


def same_padding_conv(x, w, b, s):
    out_h = math.ceil(x.size(2) / s[0])
    out_w = math.ceil(x.size(3) / s[1])

    pad_h = max((out_h - 1) * s[0] + w.size(2) - x.size(2), 0)
    pad_w = max((out_w - 1) * s[1] + w.size(3) - x.size(3), 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    x = F.conv2d(x, w, b, stride=s)
    return x


class SameConv2d(nn.Conv2d):
    """
    write by:
    https://github.com/zjjMaiMai/TinyHITNet
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return same_padding_conv(x, self.weight, self.bias, self.stride)


class ResBlock(nn.Module):
    # page 18 Fig 11
    # used in HITNet
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.conv(input)
        x = x + input
        x = self.relu(x)
        return x

######################################################################
# ********************* utilization in CFNet  **********************
# CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching
#
# ####################################################################

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self, x):
        #save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x *( torch.tanh(F.softplus(x)))


if __name__ == "__main__":
    pass