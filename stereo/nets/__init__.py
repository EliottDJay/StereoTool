from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.logger import Logger as Log
from utils.args_dictionary import get_key
from stereo.nets.CoEx import CoEx
from stereo.nets.CoEx_Original import CoEx_Original
from stereo.nets.FastACV import FastACV_Original
from stereo.nets.FastACV_Plus import FastACV_Plus_Original
from stereo.nets.HITNet_KITTI import HITNet_KITTI

STEREO_DICT = {
    'coex': CoEx,
    'coex_original': CoEx_Original,
    'fastacv':FastACV_Original,
    'fastacvplus': FastACV_Plus_Original,
    'hitnet_kitti': HITNet_KITTI
}


def model_manager(args):
    stereo_method = get_key(args, 'net', 'method').lower()

    if stereo_method not in STEREO_DICT:
        Log.error('Model: {} not valid!'.format(stereo_method))
        exit(1)

    model = STEREO_DICT[stereo_method](args)

    return model