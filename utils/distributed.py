import torch
import torch.nn as nn
import subprocess
import sys
import os
import torch.distributed as dist
from utils.logger import Logger as Log


def is_distributed():
    return torch.distributed.is_initialized()


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def all_reduce_numpy(array):
    tensor = torch.from_numpy(array).cuda()
    torch.distributed.all_reduce(tensor)
    return tensor.cpu().numpy()


def all_reduce_mean(tensor):
    tensor = tensor.clone().detach()
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        wordsize = get_world_size()
        tensor = tensor/wordsize

    return tensor

# U2PL
def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size