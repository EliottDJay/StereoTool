import logging
import os
import random
from collections import OrderedDict
import json
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from utils.logger import Logger as Log


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing


def save_args(args, path, filename='args.json'):
    # args_dict = vars(args)
    check_path(path)
    save_path = os.path.join(path, filename)

    with open(save_path, 'w') as f:
        json.dump(args, f, indent=4, sort_keys=False)


def save_command(save_path, filename='command_train.txt'):
    check_path(save_path)
    command = sys.argv
    save_file = os.path.join(save_path, filename)
    with open(save_file, 'w') as f:
        f.write(' '.join(command))


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def isNum(n):
    try:
        n=eval(n)
        if type(n)==int or type(n)==float or type(n)==complex:
            return True
    except NameError:
        return False


def count_parameters(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num


def input2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    elif isNum(v):
        if int(v) == 1:
            return True
        elif int(v) == 0:
            return False
        else:
            return False
    else:
        Log.error("Can not return a bool")
        exit(1)

