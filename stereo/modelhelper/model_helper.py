import os
import random
from collections import OrderedDict
from glob import glob

from utils.distributed import get_rank, get_world_size
import torch
import torch.distributed as dist

# AANet
def specific_params_group(kv, specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']):
    for name in specific_layer_name:
        if name in kv[0]:
            return True
    return False


def basic_params_group(kv, specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']):
    for name in specific_layer_name:
        if name in kv[0]:
            return False
    return True

# ContraSeg
"""
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        fcn_lr = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' in key:
                bb_lr.append(value)
            elif 'aux_layer' in key or 'upsample_proj' in key:
                fcn_lr.append(value)
            else:
                nbb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': fcn_lr, 'lr': self.configer.get('lr', 'base_lr') * 10},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params
"""


def load_state(path, model, optimizer=None, key="state_dict"):
    rank = get_rank()

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    if rank == 0:
                        print(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            )
                        )

        for k in ignore_keys:
            checkpoint.pop(k)

        model.load_state_dict(state_dict, strict=False)

        if rank == 0:
            ckpt_keys = set(state_dict.keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))

            metric_now = checkpoint["epe"]
            best_metric = checkpoint["best_epe"]
            last_epoch = checkpoint["epoch"]
            best_epoch = checkpoint["best_epe"]

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if rank == 0:
                print(
                    "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                        path, last_epoch
                    )
                )
        return last_epoch, best_metric, best_epoch
    else:
        if rank == 0:
            print("=> no checkpoint found at '{}'".format(path))


def save_checkpoint(save_path, optimizer, model,
                    epoch, epe, best_epe, best_epoch, filename=None, save_optimizer=True, net_name=None):

    if (net_name is None) and (filename is None):
        if hasattr(model, 'get_name'):
            net_name = model.get_name()
        else:
            net_name = 'Stereo'

    if filename is None:
        net_filename = net_name + '_epoch_{:0>3d}_'.format(epoch)
        net_filename = net_filename + '.pth'
    elif filename is not None:
        net_filename = filename

    if not net_filename.endswith('.pth'):
        net_filename = net_filename + '.pth'

    net_save_path = os.path.join(save_path, net_filename)

    if save_optimizer:
        state = {
            'epoch': epoch,
            # 'num_iter': num_iter,
            'epe': epe,
            'best_epe': best_epe,
            'best_epoch': best_epoch,
            'state_dict': model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
    elif not save_optimizer:
        state = {
            'epoch': epoch,
            # 'num_iter': num_iter,
            'epe': epe,
            'best_epe': best_epe,
            'best_epoch': best_epoch,
            'state_dict': model.state_dict(),
        }

    torch.save(state, net_save_path)


def resume_latest_ckpt(checkpoint_dir, net, optimizer, best=False):
    if best:
        ckpts = sorted(glob(checkpoint_dir + '/' + '*best.pth'))
    else:
        ckpts = sorted(glob(checkpoint_dir + '/' + '*.pth'))

    if len(ckpts) == 0:
        raise RuntimeError('=> No checkpoint found while resuming training')

    latest_ckpt = ckpts[-1]

    if hasattr(net, 'get_name'):
        net_name = net.get_name()
    else:
        net_name = 'Stereo'

    print('=> Resume latest %s checkpoint: %s' % (net_name, os.path.basename(latest_ckpt)))

    last_epoch, best_metric, best_epoch = load_state(latest_ckpt, net, optimizer)

    return last_epoch, best_metric, best_epoch
