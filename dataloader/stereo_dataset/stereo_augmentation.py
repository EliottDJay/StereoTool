from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random
from utils.logger import Logger as Log


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        left = np.transpose(sample['left'], (2, 0, 1))  # [3, H, W]
        sample['left'] = torch.from_numpy(left) / 255.
        right = np.transpose(sample['right'], (2, 0, 1))
        sample['right'] = torch.from_numpy(right) / 255.

        # disp = np.expand_dims(sample['disp'], axis=0)  # [1, H, W]
        if 'disp' in sample.keys():
            disp = sample['disp']  # [H, W]
            sample['disp'] = torch.from_numpy(disp)

        if 'pseudo_disp' in sample.keys():
            disp = sample['pseudo_disp']  # [H, W]
            sample['pseudo_disp'] = torch.from_numpy(disp)

        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['left', 'right']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample


class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        ori_height, ori_width = sample['left'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['left'] = np.lib.pad(sample['left'],
                                        ((top_pad//2, top_pad-top_pad//2), (right_pad - right_pad//2, right_pad//2), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            sample['right'] = np.lib.pad(sample['right'],
                                         ((top_pad//2, top_pad-top_pad//2), (right_pad - right_pad//2, right_pad//2), (0, 0)),
                                         mode='constant',
                                         constant_values=0)
            if 'disp' in sample.keys():
                sample['disp'] = np.lib.pad(sample['disp'],
                                            ((top_pad//2, top_pad-top_pad//2), (right_pad - right_pad//2, right_pad//2)),
                                            mode='constant',
                                            constant_values=0)

            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = np.lib.pad(sample['pseudo_disp'],
                                                   ((top_pad//2, top_pad-top_pad//2), (right_pad - right_pad//2, right_pad//2)),
                                                   mode='constant',
                                                   constant_values=0)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['left'] = self.crop_img(sample['left'])
            sample['right'] = self.crop_img(sample['right'])
            if 'disp' in sample.keys():
                sample['disp'] = self.crop_img(sample['disp'])
            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = self.crop_img(sample['pseudo_disp'])

        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]


class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, sample):
        if np.random.random() < self.p:
            sample['left'] = np.copy(np.flipud(sample['left']))
            sample['right'] = np.copy(np.flipud(sample['right']))

            sample['disp'] = np.copy(np.flipud(sample['disp']))

            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = np.copy(np.flipud(sample['pseudo_disp']))

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        sample['left'] = Image.fromarray(sample['left'].astype('uint8'))
        sample['right'] = Image.fromarray(sample['right'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['left'] = np.array(sample['left']).astype(np.float32)
        sample['right'] = np.array(sample['right']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __init__(self, p_op=1, p_diff=0.5):
        """

        :param p_op: conduct the operation
        :param p_diff: obtain two different contrast values
        """
        self.p_op = p_op
        self.p_diff = p_diff

    def __call__(self, sample):
        if np.random.random() < self.p_op:
            if np.random.random() < self.p_diff:
                contrast_factor = np.random.uniform(0.8, 1.2, 2)

                sample['left'] = F.adjust_contrast(sample['left'], contrast_factor[0])
                sample['right'] = F.adjust_contrast(sample['right'], contrast_factor[1])
            else:
                contrast_factor = np.random.uniform(0.8, 1.2)

                sample['left'] = F.adjust_contrast(sample['left'], contrast_factor)
                sample['right'] = F.adjust_contrast(sample['right'], contrast_factor)

        return sample


class RandomGamma(object):

    def __init__(self, p_op=1, p_diff=0.5):
        """

        :param p_op: conduct the operation
        :param p_diff: obtain two different contrast values
        """
        self.p_op = p_op
        self.p_diff = p_diff

    def __call__(self, sample):
        if np.random.random() < self.p_op:
            if np.random.random() < self.p_diff:
                gamma = np.random.uniform(0.8, 1.2, 2)  # adopted from FlowNet

                sample['left'] = F.adjust_gamma(sample['left'], gamma[0])
                sample['right'] = F.adjust_gamma(sample['right'], gamma[1])
            else:
                gamma = np.random.uniform(0.8, 1.2)  # adopted from FlowNet

                sample['left'] = F.adjust_gamma(sample['left'], gamma)
                sample['right'] = F.adjust_gamma(sample['right'], gamma)


        return sample


class RandomBrightness(object):

    def __init__(self, p_op=1, p_diff=0.5):
        """

        :param p_op: conduct the operation
        :param p_diff: obtain two different contrast values
        """
        self.p_op = p_op
        self.p_diff = p_diff

    def __call__(self, sample):
        if np.random.random() < self.p_op:
            if np.random.random() < self.p_diff:
                brightness = np.random.uniform(0.5, 2.0, 2)

                sample['left'] = F.adjust_brightness(sample['left'], brightness[0])
                sample['right'] = F.adjust_brightness(sample['right'], brightness[1])
            else:
                brightness = np.random.uniform(0.5, 2.0)

                sample['left'] = F.adjust_brightness(sample['left'], brightness)
                sample['right'] = F.adjust_brightness(sample['right'], brightness)

        return sample


class RandomHue(object):

    def __init__(self, p_op=1, p_diff=0.5):
        """

        :param p_op: conduct the operation
        :param p_diff: obtain two different contrast values
        """
        self.p_op = p_op
        self.p_diff = p_diff

    def __call__(self, sample):
        if np.random.random() < self.p_op:
            if np.random.random() < self.p_diff:
                hue = np.random.uniform(-0.1, 0.1, 2)

                sample['left'] = F.adjust_hue(sample['left'], hue[0])
                sample['right'] = F.adjust_hue(sample['right'], hue[1])
            else:
                hue = np.random.uniform(-0.1, 0.1)

                sample['left'] = F.adjust_hue(sample['left'], hue)
                sample['right'] = F.adjust_hue(sample['right'], hue)

        return sample


class RandomSaturation(object):

    def __init__(self, p_op=1, p_diff=0.5):
        """

        :param p_op: conduct the operation
        :param p_diff: obtain two different contrast values
        """
        self.p_op = p_op
        self.p_diff = p_diff

    def __call__(self, sample):
        if np.random.random() < self.p_op:
            if np.random.random() < self.p_op:
                saturation = np.random.uniform(0.8, 1.2, 2)

                sample['left'] = F.adjust_saturation(sample['left'], saturation[0])
                sample['right'] = F.adjust_saturation(sample['right'], saturation[1])
            else:
                saturation = np.random.uniform(0.8, 1.2)

                sample['left'] = F.adjust_saturation(sample['left'], saturation)
                sample['right'] = F.adjust_saturation(sample['right'], saturation)

        return sample


class RandomColor(object):

    def __init__(self, color_cfg):
        self.p = color_cfg['p']
        self.transform = []
        if color_cfg.get('contrast', False):
            self.transform.append(RandomContrast(p_op=color_cfg['contrast']['p_op'], p_diff=color_cfg['contrast']['p_diff']))
            Log.info("RandomContrast append")
        if color_cfg.get('gamma', False):
            self.transform.append(RandomGamma(p_op=color_cfg['gamma']['p_op'], p_diff=color_cfg['gamma']['p_diff']))
            Log.info("RandomGamma append")
        if color_cfg.get('brightness', False):
            self.transform.append(RandomBrightness(p_op=color_cfg['brightness']['p_op'], p_diff=color_cfg['brightness']['p_diff']))
            Log.info("RandomBrightness append")
        if color_cfg.get('hue', False):
            self.transform.append(RandomHue(p_op=color_cfg['hue']['p_op'], p_diff=color_cfg['hue']['p_diff']))
            Log.info("RandomHue append")
        if color_cfg.get('saturation', False):
            self.transform.append(RandomSaturation(p_op=color_cfg['saturation']['p_op'], p_diff=color_cfg['saturation']['p_diff']))
            Log.info("RandomSaturation append")


    def __call__(self, sample):

        sample = ToPILImage()(sample)
        if np.random.random() < self.p:
            # A single transform
            t = random.choice(self.transform)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(self.transform)
            for t in self.transform:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample


class GaussNoise(object):

    def __init__(self, limit_min=10, limit_max=50):
        self.min = limit_min
        self.max = limit_max

    def __call__(self, sample):
        var_limit = (self.min, self.max)
        mean = 0
        if np.random.random() < 0.5:
            var = random.uniform(var_limit[0], var_limit[1])
            sigma = var ** 0.5
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

            gauss = random_state.normal(mean, sigma, sample['left'].shape)
            sample['left'] = gauss + sample['left']
            sample['left'] = np.clip(sample['left'], 0., 255.)

        if np.random.random() < 0.5:
            var = random.uniform(var_limit[0], var_limit[1])
            sigma = var ** 0.5
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

            gauss = random_state.normal(mean, sigma, sample['right'].shape)
            sample['right'] = gauss + sample['right']
            sample['right'] = np.clip(sample['right'], 0., 255.)

        return sample


class RGBShift(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            r_shift = random.uniform(-20, 20)
            g_shift = random.uniform(-20, 20)
            b_shift = random.uniform(-20, 20)
            sample['left'] = self._shift_rgb_non_uint8(sample['left'], r_shift, g_shift, b_shift)
            sample['left'] = np.clip(sample['left'], 0., 255.)

        if np.random.random() < 0.5:
            r_shift = random.uniform(-20, 20)
            g_shift = random.uniform(-20, 20)
            b_shift = random.uniform(-20, 20)
            sample['right'] = self._shift_rgb_non_uint8(sample['right'], r_shift, g_shift, b_shift)
            sample['right'] = np.clip(sample['right'], 0., 255.)

        return sample

    def _shift_rgb_non_uint8(img, r_shift, g_shift, b_shift):
        if r_shift == g_shift == b_shift:
            return img + r_shift

        result_img = np.empty_like(img)
        shifts = [r_shift, g_shift, b_shift]
        for i, shift in enumerate(shifts):
            result_img[..., i] = img[..., i] + shift

        return result_img


class Occlude(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            sx = int(np.random.uniform(35, 100))
            sy = int(np.random.uniform(25, 75))
            cx = int(np.random.uniform(sx, sample['right'].shape[0] - sx))
            cy = int(np.random.uniform(sy, sample['right'].shape[1] - sy))
            sample['right'][cx-sx:cx+sx, cy-sy:cy+sy] = np.mean(np.mean(sample['right'], 0), 0)[np.newaxis, np.newaxis]

        return sample