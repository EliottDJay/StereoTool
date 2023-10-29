import torch
import torch.nn as nn
import torch.nn.functional as F
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


# Used for CFNet
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self, x):
        #save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x *( torch.tanh(F.softplus(x)))


class ChannelAtt(SubModule):
    def __init__(self, cv_chan, im_chan, D):
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

"""def acf_choose(active_function=None, **params):
    if active_function.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif active_function.lower() == 'leakyrelu':
        if 'alpha' not in params:
            params['alpha'] = 0.2
        return nn.LeakyReLU(params['alpha'], inplace=True)
    elif active_function.lower() == 'mish':
        return Mish()
    elif active_function.lower() == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError"""


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
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
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

