import argparse
import copy

import numpy as np
import os
import time
import random
import yaml
from datetime import datetime
import sys

current_path = os.path.abspath(__file__)
file_split = current_path.split('/')

path_new = os.path.join(* file_split[:-2] )
abspath = "/" + path_new
sys.path.append(abspath)

from utils import utils
from utils.average_meter import AverageMeter
from utils.logger import Logger as Log
from utils.distributed import setup_distributed, is_distributed, all_reduce_mean, get_world_size, get_rank
from utils.stereo_utils.stereo_visualization import save_images, disp_error_img
from utils.stereo_utils.stereo_metrics import d1_metric, thres_metric

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from stereo.nets import model_manager
from stereo.modelhelper.losshelper import loss_helper
from dataloader.stereo_dataset.stereo_loader import get_loader
from stereo.modelhelper.optim_scheduler import get_scheduler, get_optimizer
from stereo.modelhelper.model_helper import load_state, specific_params_group, basic_params_group, \
    resume_latest_ckpt, save_checkpoint


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def int2bool(v):
    """ Usage:
    parser.add_argument('--x', type=int2bool, nargs='?', const=True,
                        dest='x', help='Whether to use pretrained models.')
    """
    if int(v) == 1:
        return True
    elif int(v) == 0:
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser(description='Train stereo network')

    # if 'LOCAL_RANK' not in os.environ:
    #       os.environ['LOCAL_RANK'] = str(args.local_rank)

    parser.add_argument('--configs',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('--seed', type=int, default=326)  # 0
    # parser.add_argument('--gpu', default=[1, 3], nargs='+', type=int, dest='gpu', help='The gpu list used.')  # 这个还有用吗
    parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')
    parser.add_argument('--deterministic', type=str2bool, nargs='?', default=True, help='Use CUDNN.')

    # ***********  Params for distributed training.  **********
    # parser.add_argument('--local-rank', type=int, default=0, help='local rank of current process')
    parser.add_argument('--distributed', type=int2bool, default=True,
                        help='Use multi-processing training.')
    parser.add_argument('--port', type=int, default=29961, help='.')

    # ***********  Params for experiment and checkpoint.  **********
    parser.add_argument('--yaml_path', type=str, default=None, help='.')
    # parser.add_argument('--resume', default=False, help='Resume training from latest checkpoint')
    # parser.add_argument('--pretrained_net', default=None, type=str, help='Pretrained network')
    # parser.add_argument('--exp_dir', default="/data/data2/drj/ML/Stereo/StereoTool/checkpoint/main_expi1/sub_expi1")

    # ***********  Params for logging. and screen  **********
    parser.add_argument('--logfile_level', default='info', type=str, help='To set the log level to files.')
    parser.add_argument('--stdout_level', default='info', type=str, help='To set the level to print to screen.')
    # parser.add_argument('--log_file', default="log/stereo.log", type=str, dest='logging:log_file', help='The path of log files.')
    parser.add_argument('--rewrite', type=str2bool, nargs='?', default=False, help='Whether to rewrite files.')
    parser.add_argument('--log_to_file', type=str2bool, nargs='?', default=True, help='Whether to write logging into files.')
    parser.add_argument('--log_format', type=str, nargs='?', default="%(asctime)s %(levelname)-7s %(message)s"
                        , help='Whether to write logging into files.')

    # ***********  Extra para related to training mode  **********
    parser.add_argument('--evaluate_only', default=False, help='Only evaluate pretrained models')
    parser.add_argument('--freeze_bn', default=False, help='Switch BN to eval mode to fix running statistics')
    parser.add_argument('--print_freq', default=100, type=int, help='Print frequency to screen (iterations)')
    parser.add_argument('--summary_freq', default=100, type=int,
                        help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=1, type=int,
                        help='Save checkpoint frequency (epochs)')  # For SenceFlow 1

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser = parser.parse_args()

    cfg = yaml.load(open(parser.configs, "r"), Loader=yaml.FullLoader)

    parser = vars(parser)
    parser_new = parser.copy()
    parser_new.update(cfg)

    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        gpu = copy.deepcopy(os.environ.get('CUDA_VISIBLE_DEVICES'))
        gpu1 = gpu.split(",")
        gpulist = [int(i.strip()) for i in gpu1]
        parser_new["gpu"] = gpulist

    if parser_new["seed"] is not None:
        os.environ['PYTHONHASHSEED'] = str(parser_new["seed"])  # 为了禁止hash随机化，使得实验可复现

    return parser_new


def main():
    args = parse_args()

    """
    - experiment: bash file to run the experiment
    - config: config file related to the experiment
    - checkpoint : store the checkpoint log, summary file during training or testing. It is a main experiment set
        - main_expi1:
            - sub_expi1    (args['exp_dir']) : may be experiment on SceneFlow
                - log_file (args["log_file"]):
                - summary:
                - model    (args["model_path"]):
                lastest_model
                best_model
                args
                val_results.txt
            - sub_expi2: finetuning experiment
        - main_expi2:  experiments with absolutely new config
    """

    if args.get("yaml_path", False):
        checkpoint_root = "/data/data2/drj/ML/Stereo/StereoTool/checkpoint/"
        yaml_path = args["yaml_path"]
        path_split = yaml_path.split('/')
        args["exp_dir"] = os.path.join(checkpoint_root, *path_split[-3:])

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = args['exp_dir']  # os.path.dirname(args["config"])
    args["model_path"] = os.path.join(exp_path, "model")
    args["log_file"] = os.path.join(exp_path, 'log')
    summary_path = os.path.join(exp_path, "summary")
    utils.check_path(args["model_path"])
    utils.check_path(args["log_file"])
    utils.check_path(summary_path)
    args["log_file"] = os.path.join(exp_path, 'log', "stereo" + current_time + ".log")

    cudnn.enabled = True
    cudnn.benchmark = True

    rank = get_rank()

    if args["seed"] is not None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        # https://blog.csdn.net/qq_42714262/article/details/121722064
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])
        torch.cuda.manual_seed_all(args["seed"])
    if args['deterministic']:
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
        # torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
        # RuntimeError: upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation,
        # but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation,
        # or you can use the 'warn_only=True' option, if that's acceptable for your application.
        # You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
        cudnn.deterministic = True
        torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
        cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。

    # torch.autograd.set_detect_anomaly(True)

    if rank == 0:
        Log.init(logfile_level=args['logfile_level'],
                 stdout_level=args['stdout_level'],
                 log_file=args["log_file"],
                 log_format=args['log_format'],
                 rewrite=args['rewrite'])
        # Log.info("{}".format(pprint.pformat(args)))
        utils.save_args(args, args["exp_dir"])
        utils.save_command(args["exp_dir"], "command.txt")
        tb_logger = SummaryWriter(summary_path)
    else:
        tb_logger = None

    # Create network.
    model = model_manager(args)
    model_name = model.get_name()
    num_params_train, num = utils.count_parameters(model)
    if rank == 0:
        Log.info('=> Number of trainable parameters: %d' % num_params_train)
        Log.info('=> Number of parameters: %d' % num)
        open(os.path.join(exp_path, '%d_parameters_trainable' % num_params_train), 'a').close()
        open(os.path.join(exp_path, '%d_parameters' % num), 'a').close()

    if args["net"]["sync_bn"] and args["distributed"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    if args["distributed"]:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        )
    else:
        model = torch.nn.DataParallel(model).cuda()

    loss = loss_helper(args)
    loss.cuda()  # need this?

    if not args["dataset"].get("mode", False):
        if args["dataset"]["type"] == "KITTI_Mix":
            mode = "noval"
        else:
            mode = "val"
    else:
        mode = args["dataset"]["mode"]
    train_loader, val_loader = get_loader(args, mode, args["seed"])

    # AAnet: Learning rate for offset learning is set 0.1 times those of existing layers
    specific_params = list(filter(specific_params_group, model.named_parameters()))
    base_params = list(filter(basic_params_group, model.named_parameters()))
    specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
    base_params = [kv[1] for kv in base_params]
    basic_lr = args["trainer"]["optimizer"]["kwargs"]["lr"]
    specific_lr = basic_lr * 0.1
    params_group = [
        {'params': base_params, 'lr': basic_lr},
        {'params': specific_params, 'lr': specific_lr},
    ]

    cfg_trainer = args["trainer"]
    optimizer = get_optimizer(params_group, cfg_trainer)

    best_pred = 999.0
    start_epoch = 0
    best_epoch = 0
    pred = 0
    all_epoch = cfg_trainer['epochs']
    scheduler_metric = args['trainer']['lr_scheduler'].get('metric', 'epoch')
    train_metric = args['trainer'].get('metric', 'epoch')
    if train_metric == "iter":
        args["save_ckpt_freq"] = args["save_ckpt_freq"] * len(train_loader)  # iter

    # auto_resume > pretrain
    if args['resume']:
        start_epoch, best_pred, best_epoch = resume_latest_ckpt(args["exp_dir"], model, optimizer)
        # last_epoch, best_prec, best_epoch = resume_latest_ckpt os.path.join(args["exp_dir"], "ckpt.pth")
    elif args["pretrained_net"] is not None:
        # usually for fine turning
        load_state(args["pretrained_net"], model)

    # AANet-style: have a mode only conduct evaluate
    if args['evaluate_only']:
        assert args['dataset']['val']['batch_size'] == 1
        validate(model, val_loader, 1, args, tb_logger, best_pred)
        Log.info("Validation process down in")
        exit(1)

    if not args['resume']:
        files = os.listdir(exp_path)
        for file in files:
            if file.endswith('.pth'):
                Log.error("Experiment exit.\
                          The purpose of this error message is to prevent overwriting the original experiment!")
                exit(1)

    # last_epoch = last_epoch if args['resume'] else last_epoch - 1 ?
    last_epoch = start_epoch if args['resume'] else start_epoch - 1
    scheduler = get_scheduler(cfg_trainer, optimizer, len(train_loader), last=last_epoch)

    # if not args['evaluate_only']:
    epoch = start_epoch
    for _ in range(start_epoch, all_epoch):
        epoch = train(model, optimizer, scheduler, loss, train_loader, epoch, train_metric, args, scheduler_metric, tb_logger)
        if train_metric == 'epoch':
            epoch = epoch + 1

        if mode != 'noval':
            # validate cant be with rank=0. All_reduce may wait for response from the sub-process from all ranks
            pred = validate(model, val_loader, epoch, args, tb_logger, best_pred)

        if rank == 0:
            if mode != 'noval':
                filename = model_name + '_best.pth'
                if pred < best_pred:
                    best_pred = pred
                    best_epoch = epoch
                    save_checkpoint(args["exp_dir"], optimizer, model, epoch, pred, best_pred, best_epoch, filename=filename)

                if epoch == all_epoch:
                    val_file = os.path.join(args['exp_dir'], 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('\nbest epoch: %03d \t best EPE: %.3f\n\n' % (best_epoch, best_pred))

                    Log.info('=> best epoch: %03d \t best EPE: %.3f\n' % (best_epoch, best_pred))

            if mode == 'noval':
                best_pred, pred, best_epoch = -1, -1, -1

            filename = model_name + '_latest.pth'
            save_checkpoint(args["exp_dir"], optimizer, model, epoch, pred, best_pred, best_epoch, filename=filename)

            if epoch % args["save_ckpt_freq"] == 0:
                save_checkpoint(args["model_path"], optimizer, model, epoch, pred, best_pred, best_epoch, save_optimizer=False, net_name=model_name)

        if scheduler_metric == 'epoch':
            base_lr = optimizer.param_groups[0]['lr']
            if rank == 0:
                tb_logger.add_scalar('base_lr', base_lr, epoch)

            scheduler.step()

        if epoch == all_epoch:
            break


def train(model, optimizer, lr_scheduler, stereo_loss, data_loader, epoch, train_metric, args, scheduler_metric, tb_logger):
    model.train()
    # stereo_loss.train()
    if args["freeze_bn"]:
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        model.apply(set_bn_eval)
    if data_loader.sampler is not None and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)

    rank = get_rank()

    # to screen
    total_iteration = len(data_loader)
    total_epoch = args['trainer']['epochs']  # all epoch
    print_freq = args.get('print_freq', 100)
    summary_freq = args.get('summary_freq', 100)
    max_disp = args['dataset']['max_disparity']
    train_metric = args['trainer'].get('metric', 'epoch')
    if train_metric == "epoch":
        pre_iter = epoch * total_iteration
    elif train_metric == "iter":
        pre_iter = epoch

    pseudo_gt = args['dataset']['train'].get('pseudo_gt', False)
    slant = args['dataset']['train'].get('slant', False)

    # scaler = torch.cuda.amp.GradScaler()

    disp_losses = AverageMeter()
    total_losses = AverageMeter()
    batch_time = AverageMeter()
    forward_time = AverageMeter()
    # backward_time = AverageMeter()
    # loss_time = AverageMeter()
    data_time = AverageMeter()

    for step in range(total_iteration):
        # torch.cuda.synchronize()
        i_iter = pre_iter + step + 1

        data_start = time.time()
        sample = next(data_loader_iter)

        img_left = sample['left'].cuda()  # [B, 3, H, W]
        img_right = sample['right'].cuda()
        gt_disp = sample['disp'].cuda()  # [B, H, W

        mask = (gt_disp > 1e-3) & (gt_disp < max_disp)# ]

        if pseudo_gt:
            pseudo_gt_disp = sample['pseudo_disp'].cuda()
        else:
            pseudo_gt_disp = None

        if slant:
            dxdy = sample["dxdy"].cuda()
        else:
            dxdy = None

        data_time.update(time.time() - data_start)

        forward_start = time.time()
        # with torch.autograd.detect_anomaly():
        # with torch.cuda.amp.autocast():
        preds_dic = model(img_left, img_right)
        forward_time.update(time.time() - forward_start)
        # assert not torch.any(torch.isnan(preds[-1]))

        loss_dic = stereo_loss(preds_dic, gt_disp, dxygt=dxdy, pseudo_disp=pseudo_gt_disp,)

        # assert not torch.any(torch.isnan(total_loss))
        """
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        """

        total_loss = loss_dic["total_loss"]

        optimizer.zero_grad()
        total_loss.backward()
        #  torch.isnan(model.module.mu).sum() == 0, print(model.module.mu)
        optimizer.step()

        # assert torch.isnan(model.module.mu).sum() == 0, print(model.module.mu)
        # assert torch.isnan(model.module.mu.grad).sum() == 0, print(model.module.mu.grad)

        """
        for name, param in model.named_parameters():
                    if param.grad is None:
                        print(name)
                        """

        batch_time.update(time.time() - data_start)

        # get the disp loss
        for name in loss_dic.keys():
            if ("multi_preds" in name) and ("pyramid" not in name):
                disp_loss = loss_dic[name]

        # gather all loss from different gpus
        reduced_total_loss = all_reduce_mean(total_loss)
        reduced_disp_loss = all_reduce_mean(disp_loss)
        total_losses.update(reduced_total_loss)
        disp_losses.update(reduced_disp_loss)

        if rank == 0 and (i_iter % print_freq == 0):
            Log.info(
                "Epoch: [{}/{}]  Iter: [{}/{}] /all {} \t"
                "Data Time: {data_time.val:.2f} ({data_time.avg:.2f}) "
                "Forward Time: {forward_time.val:.2f} ({forward_time.avg:.2f})\t"
                "Batch Time: {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Disp Loss: {disp_loss.val:.4f} ({disp_loss.avg:.4f}) "
                "Total Loss: {total_loss.val:.4f} ({total_loss.avg:.4f})\t".format(
                    epoch + 1, total_epoch, step+1, total_iteration, i_iter,
                    data_time=data_time,
                    forward_time=forward_time,
                    batch_time=batch_time,
                    disp_loss=disp_losses,
                    total_loss=total_losses
                            )
            )

        # EPE and error point count
        # get the disp
        for name in preds_dic.keys():
            if ("preds" in name) and ("pyramid" in name):
                preds = preds_dic[name]

        pred_disp = preds[-1]
        if pred_disp.size(-1) != gt_disp.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                      mode='bilinear', align_corners=False) * (
                                gt_disp.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
        thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
        epe = all_reduce_mean(epe)
        d1 = all_reduce_mean(d1)
        thres1 = all_reduce_mean(thres1)
        thres2 = all_reduce_mean(thres2)
        thres3 = all_reduce_mean(thres3)

        # most related to the image summary
        if rank == 0 and (i_iter % summary_freq == 0):
            img_summary = dict()
            img_summary['left'] = img_left
            img_summary['right'] = img_right
            img_summary['gt_disp'] = gt_disp

            if pseudo_gt:
                img_summary['pseudo_gt_disp'] = pseudo_gt_disp

            # Save pyramid disparity prediction
            for s in range(len(preds)):
                # Scale from low to high, reverse
                # Maybe only save image at one GPU (Rank = 0)
                save_name = 'pred_disp' + str(len(preds) - s - 1)
                save_value = preds[s]
                img_summary[save_name] = save_value

            img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
            save_images(tb_logger, 'train', img_summary, i_iter)

            tb_logger.add_scalar('train/epe', epe.item(), i_iter)
            tb_logger.add_scalar('train/d1', d1.item(), i_iter)
            tb_logger.add_scalar('train/disp_loss', disp_losses.avg, i_iter)
            if pseudo_gt:
                tb_logger.add_scalar('train/total_loss', total_losses.avg, i_iter)

            tb_logger.add_scalar('train/thres1', thres1.item(), i_iter)
            tb_logger.add_scalar('train/thres2', thres2.item(), i_iter)
            tb_logger.add_scalar('train/thres3', thres3.item(), i_iter)

            stereo_loss.loss_tb_logger(tb_logger, loss_dic, i_iter)

            disp_losses.reset()
            total_losses.reset()
            batch_time.reset()
            forward_time.reset()
            # backward_time.reset()
            # loss_time.reset()
            data_time.reset()

        if train_metric == 'iter':
            epoch = i_iter
            if epoch >= total_epoch:
                return epoch

        if scheduler_metric == 'iter':
            base_lr = optimizer.param_groups[0]['lr']
            tb_logger.add_scalar('base_lr', base_lr, i_iter)
            lr_scheduler.step()

    return epoch

    # Always save the latest model for resuming training

def validate(model, data_loader, epoch, args, tb_logger, best_prec):
    model.eval()

    if data_loader.sampler is not None and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(1)

    Log.info('=> Start validation...')

    validate_screen_freq = 5

    rank = get_rank()
    word_size = get_world_size()

    max_disp = args['dataset']['max_disparity']

    data_loader_iter = iter(data_loader)
    num_samples = len(data_loader)

    freq = num_samples // validate_screen_freq

    Log.info('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres1 = 0
    val_thres2 = 0
    val_thres3 = 0
    val_count = 0
    num_imgs = 0
    valid_samples = 0

    for step in range(num_samples):
        if step % freq == 0:
            Log.info('=> Validating %d/%d' % (step, num_samples))

        sample = next(data_loader_iter)

        img_left = sample['left'].cuda()  # [1, 3, H, W]
        img_right = sample['right'].cuda()
        gt_disp = sample['disp'].cuda()  # [1, H, W]
        mask = (gt_disp > 0) & (gt_disp < max_disp)

        if not mask.any():
            continue

        num_imgs += gt_disp.size(0)

        with torch.no_grad():
            pred_dic = model(img_left, img_right)  # not [B, H, W] but dist

        # get the disp
        for name in pred_dic.keys():
            if ("preds" in name) and ("pyramid" in name):
                pred_disp = pred_dic[name][-1]

        if pred_disp.size(-1) < gt_disp.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, (gt_disp.size(-2), gt_disp.size(-1)),
                                      mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')

        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
        thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)

        epe = all_reduce_mean(epe)
        d1 = all_reduce_mean(d1)
        thres1 = all_reduce_mean(thres1)
        thres2 = all_reduce_mean(thres2)
        thres3 = all_reduce_mean(thres3)

        if rank == 0:
            # Sum operation when rank = 0
            val_epe += epe.item()
            val_d1 += d1.item()
            val_thres1 += thres1.item()
            val_thres2 += thres2.item()
            val_thres3 += thres3.item()
            # val_count += 1
            valid_samples += 1

            # Save 3 images for visualization
            if not args['evaluate_only']:
                if step in [num_samples // 4, num_samples // 2, num_samples // 4 * 3]:
                    img_summary = dict()
                    img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
                    img_summary['left'] = img_left
                    img_summary['right'] = img_right
                    img_summary['gt_disp'] = gt_disp
                    img_summary['pred_disp'] = pred_disp
                    save_images(tb_logger, 'val' + str(val_count), img_summary, epoch)
                    val_count += 1

    if rank == 0:
        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples
        mean_thres1 = val_thres1 / valid_samples
        mean_thres2 = val_thres2 / valid_samples
        mean_thres3 = val_thres3 / valid_samples
        Log.info('=> Validation done!')

        val_file = os.path.join(args['exp_dir'], 'val_results.txt')
        # Save validation results
        with open(val_file, 'a') as f:
            f.write('epoch: %03d\t' % epoch)
            f.write('epe: %.3f\t' % mean_epe)
            f.write('d1: %.4f\t' % mean_d1)
            f.write('thres1: %.4f\t' % mean_thres1)
            f.write('thres2: %.4f\t' % mean_thres2)
            f.write('thres3: %.4f\n' % mean_thres3)

        Log.info('=> Mean validation epe of epoch %d: %.3f' % (epoch, mean_epe))

        if not args['evaluate_only']:
            tb_logger.add_scalar('val/epe', mean_epe, epoch)
            tb_logger.add_scalar('val/d1', mean_d1, epoch)
            tb_logger.add_scalar('val/thres1', mean_thres1, epoch)
            tb_logger.add_scalar('val/thres2', mean_thres2, epoch)
            tb_logger.add_scalar('val/thres3', mean_thres3, epoch)

        return mean_epe
    return None  # rank != 0

if __name__ == "__main__":
    main()





