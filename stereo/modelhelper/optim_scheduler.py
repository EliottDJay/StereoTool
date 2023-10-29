from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
from utils.logger import Logger as Log

import torchcontrib
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def get_optimizer(parms, cfg_trainer):
    # Get the optimizer
    cfg_optim = cfg_trainer["optimizer"]
    optim_type = cfg_optim["type"]
    optim_kwargs = cfg_optim["kwargs"]
    if optim_type == "SGD":
        optimizer = SGD(parms, **optim_kwargs)
    elif optim_type == "Adam":
        optimizer = Adam(parms, **optim_kwargs)
    elif optim_type == "AdamW":
        optimizer = AdamW(parms, **optim_kwargs)
    else:
        optimizer = None

    assert optimizer is not None, "optimizer type is not supported by LightSeg"

    return optimizer


def get_scheduler(cfg_trainer, optimizer, len_data, last):
    scheduler = None
    cfg_lr = cfg_trainer["lr_scheduler"]
    policy = cfg_lr["mode"]
    metric = cfg_lr["metric"]
    lr_kwargs = cfg_lr["kwargs"]
    max_iters = int(cfg_trainer["epochs"] * len_data)
    if policy == 'step':
        if metric == 'epoch':
            scheduler = lr_scheduler.StepLR(optimizer, **lr_kwargs, last_epoch=last)
        else:
            Log.error('lr_scheduler.StepLR cant update following iter updating but the epoch!')
            exit(1)
    elif policy == 'multistep':
        if metric == 'epoch':
            scheduler = lr_scheduler.MultiStepLR(optimizer, **lr_kwargs, last_epoch=last)
        else:
            Log.error('lr_scheduler.MultiStepLR cant update following iter updating but the epoch!')
            exit(1)
    elif policy == 'lambda_poly':
        if metric == 'iter':
            power = lr_kwargs.get('power', 0.9)
            Log.info('Use lambda_poly policy with power {}'.format(0.9))
            lambda_poly = lambda iters: pow((1.0 - iters / max_iters), power)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly, last_epoch=last)
        else:
            Log.error('lambda_poly cant update following epoch updating but the iter!')
            exit(1)
    elif policy == 'lambda_cosine':
        if metric == 'iter':
            lambda_cosine = lambda iters: (math.cos(math.pi * iters / max_iters)
                                           + 1.0) / 2
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine, last_epoch=last)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **lr_kwargs)
    elif policy == 'swa_lambda_poly':
        optimizer = torchcontrib.optim.SWA(optimizer)
        normal_max_iters = int(max_iters * 0.75)
        swa_step_max_iters = (max_iters - normal_max_iters) // 5 + 1  # we use 5 ensembles here

        def swa_lambda_poly(iters):
            if iters < normal_max_iters:
                return pow(1.0 - iters / normal_max_iters, 0.9)
            else:  # set lr to half of initial lr and start swa
                return 0.5 * pow(1.0 - ((iters - normal_max_iters) % swa_step_max_iters) / swa_step_max_iters, 0.9)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_lambda_poly, last_epoch=last)
    elif policy == 'swa_lambda_cosine':
        optimizer = torchcontrib.optim.SWA(optimizer)
        normal_max_iters = int(max_iters * 0.75)
        swa_step_max_iters = (max_iters - normal_max_iters) // 5 + 1  # we use 5 ensembles here

        def swa_lambda_cosine(iters):
            if iters < normal_max_iters:
                return (math.cos(math.pi * iters / normal_max_iters) + 1.0) / 2
            else:  # set lr to half of initial lr and start swa
                return 0.5 * (math.cos(
                    math.pi * ((iters - normal_max_iters) % swa_step_max_iters) / swa_step_max_iters) + 1.0) / 2

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=swa_lambda_cosine, last_epoch=last)

    elif policy == 'warmup_cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1000,
                                         t_total=max_iters, last_epoch=last)

    else:
        Log.error('Policy:{} is not valid.'.format(policy))
        exit(1)

    return scheduler



