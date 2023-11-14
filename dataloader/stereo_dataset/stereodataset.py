from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import copy
import torch.distributed as dist
import random

from utils.distributed import is_distributed
from utils.utils import read_text_lines, np2torch
from utils.stereo_utils.stereo_io import read_img, read_disp
from dataloader.stereo_dataset import stereo_augmentation
from utils.logger import Logger as Log


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class StereoDataset(Dataset):
    def __init__(self, dataset_name, data_root, data_list, transform=None, load_pseudo_gt=False,
                 save_filename=False, slant=False, scale_factor=None, seed=None):
        super(StereoDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.transform = transform
        self.scale_factor = scale_factor
        self.save_filename = save_filename
        self.slant = slant

        self.samples = []
        if seed is not None:
            random.seed(seed)

        if scale_factor is None and dataset_name is 'SCARED':
            self.scale_factor = 128.

        lines = read_text_lines(data_list)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]

            sample = dict()

            if save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_root, left_img)
            sample['right'] = os.path.join(data_root, right_img)
            sample['disp'] = os.path.join(data_root, gt_disp) if gt_disp is not None else None

            if self.slant:
                if "kitti" in self.dataset_name.lower():
                    splits = str(gt_disp).split("/")
                    name = os.path.splitext(splits[-1])[0]  # get the name .npy
                    name = name + ".npy"
                    mode = splits[0]
                    tag = splits[1]
                    if "0" in tag and "disp_occ" in tag:
                        set = "_2015"
                    elif "0" not in tag and "disp_occ" in tag:
                        set = "_2012"

                    sample['dxdy'] = os.path.join(data_root, mode, ("slant_window"+set), name)

            if load_pseudo_gt and sample['disp'] is not None:
                # KITTI 2015
                if 'disp_occ_0' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ_0',
                                                                     'disp_occ_0_pseudo_gt')

                # KITTI 2012
                elif 'disp_occ' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ',
                                                                     'disp_occ_pseudo_gt')
                else:
                    raise NotImplementedError
            else:
                sample['pseudo_disp'] = None

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=subset, scale_factor=self.scale_factor)  # [H, W]
        if sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'], subset=subset)  # [H, W]

        if self.slant:
            sample["dxdy"] = read_disp(sample_path["dxdy"])  # [2, H, W]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)


def build_transfrom(cfg, mean, std, stage="train"):
    transform = []
    if cfg.get("crop", False):
        if stage == "val" or stage == "test":
            validate = True
        else:
            validate = False
        img_height, img_width = cfg["crop"]["height"], cfg["crop"]["width"]
        transform.append(stereo_augmentation.RandomCrop(img_height, img_width, validate=validate))
        Log.info("using RandomCrop to tranform {} img to height{} x width{}".format(stage, img_height, img_width))
    if cfg.get('color', False):
        color_cfg = cfg['color']
        transform.append(stereo_augmentation.RandomColor(color_cfg))
        Log.info("Using Color Transfomer to {} dataset. Detail set as below:".format(stage))
    if cfg.get('rightshift', False):
        shift_cfg = cfg['rightshift']
        transform.append(stereo_augmentation.ShiftRight(shift_cfg))
    if cfg.get('occlude', False):
        occlude_cfg = cfg['occlude']
        transform.append(stereo_augmentation.Occlude(occlude_cfg))
        Log.info("Using Occlude Transfomer to {} dataset".format(stage))
    if cfg.get('vflip', False):
        transform.append(stereo_augmentation.RandomVerticalFlip(cfg["vflip"]["p"]))
        Log.info("Using RandomVerticalFlip Transfomer to {} dataset".format(stage))
    transform.append(stereo_augmentation.ToTensor())
    transform.append(stereo_augmentation.Normalize(mean, std))
    transform = stereo_augmentation.Compose(transform)
    return transform


def build_stereoloader(dataset_cfg, stage, seed=None):
    mode_cfg = copy.deepcopy(dataset_cfg[stage])

    cfg = copy.deepcopy(dataset_cfg)
    img_mean = cfg['mean']
    img_std = cfg['std']

    dataset_name = cfg.get('type', 'SceneFlow')
    data_root = mode_cfg['dataroot']
    datalist = mode_cfg['data_list']

    workers = cfg.get("workers", 0)
    batch_size = mode_cfg.get("batch_size", 1)
    shuffle = mode_cfg.get("shuffle", False)
    memory_pin = mode_cfg.get("pin_memory", True)
    drop = mode_cfg.get("drop_last", False)
    pseudo_gt = mode_cfg.get("pseudo_gt", False)
    save_filename = mode_cfg.get("save_filename", False)
    scale_factor = mode_cfg.get("scale_factor", None)
    slant = mode_cfg.get("slant", False)

    Log.info("Collecting {} dataset for {} with {} batchsize!".format(dataset_name, stage, batch_size))
    transform = build_transfrom(mode_cfg, img_mean, img_std, stage=stage)

    stereodataset = StereoDataset(dataset_name, data_root, datalist, transform=transform, load_pseudo_gt=pseudo_gt
                                  , save_filename=save_filename, slant=slant, scale_factor=scale_factor, seed=seed)

    if is_distributed():
        sampler = DistributedSampler(stereodataset)
    else:
        sampler = None

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    loader = DataLoader(
        stereodataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=workers,
        sampler=sampler,
        pin_memory=memory_pin,
        drop_last=drop,
        worker_init_fn=seed_worker,
        generator=g
    )
    # 删除了 shuffle
    # https://blog.csdn.net/qq_41554005/article/details/114583572

    return loader

