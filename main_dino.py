# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from re import A
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import resized_crop, hflip

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    
    parser.add_argument('--given_pos', action='store_true', help='Replace cls_pos_embed with interpolated patch_pos_embed.')
    parser.add_argument('--with_cls_token', action='store_true', help='Reference token shared learnable class token.')
    parser.add_argument('--another_center', action='store_true', help='Use separate centering for given_pos_token.')
    parser.add_argument('--num_reference', default=1, type=int, help="Number of points sampled per crop. Use k*k points in actual.")
    parser.add_argument('--sampling_mode', type=str, default='random', choices=['random', 'grid'], help='Mode of reference point sampling.')
    parser.add_argument('--mask_mode', type=str, default='020', choices=['020', 'all2pos', 'all2pos_pos2cls'], help='Masked Attention.')

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, 'argument.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.num_reference,
        args.sampling_mode
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
            given_pos=args.given_pos,
            with_cls_token=args.with_cls_token
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size, given_pos=args.given_pos, with_cls_token=args.with_cls_token)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.another_center
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, ((images, tea_pos, stu_pos), _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        tea_pos = [pos.cuda(non_blocking=True) for pos in tea_pos]
        stu_pos = [pos.cuda(non_blocking=True) for pos in stu_pos]

        # sampling reference point
        # images: [(2+x) * [B,3,H,W]]
        # tea_pos: 2 * [B, 2+x, k*k, 2], stu_pos: (2+x) * [B, 2, K*k, 2]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2], tea_pos, args.mask_mode)  # only the 2 global views pass through the teacher
            student_output = student(images, stu_pos, args.mask_mode)
            loss, global_loss, ref_loss = dino_loss(student_output, teacher_output, epoch, args.given_pos)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item(), global_loss=global_loss.item(), ref_loss=ref_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, another_center=False, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.another_center = another_center
        if another_center:
            self.register_buffer("ref_center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, given_pos=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        if given_pos:
            num_ref_point = student_output.shape[1] // 2

        student_out = student_output / self.student_temp
        student_out_global = student_out[:,0].chunk(self.ncrops) # [2+x, [B, dim]]
        if given_pos:
            student_out_ref = student_out[:,1:].unflatten(1, (2,num_ref_point)).chunk(self.ncrops)   # [2+x, [B, 2, k, dim]]

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        # teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out_global = F.softmax((teacher_output[:,0] - self.center) / temp, dim=-1).detach().chunk(2)  # [2, [B, dim]]
        if given_pos:
            teacher_out_ref_temp = teacher_output[:,1:].unflatten(1, (self.ncrops,num_ref_point))  # [2B, 2+x, k, dim]
            if self.another_center:
                teacher_out_ref_temp = F.softmax((teacher_out_ref_temp - self.ref_center) / temp, dim=-1)
            else:
                teacher_out_ref_temp = F.softmax((teacher_out_ref_temp - self.center) / temp, dim=-1)
            teacher_out_ref = teacher_out_ref_temp.detach().chunk(2) # [2, [B, 2+x, k, dim]]

        global_loss, ref_loss = 0, 0
        global_n_loss_terms, ref_n_loss_terms = 0, 0

        for iq in range(2): # 2 == len(teacher_out_global)
            for v in range(self.ncrops):    # self.ncrops == len(student_out_global)
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                # global cls token loss
                loss = torch.sum(-teacher_out_global[iq] * F.log_softmax(student_out_global[v], dim=-1), dim=-1)
                global_loss += loss.mean()
                global_n_loss_terms += 1
                
                # given position token loss
                if given_pos:
                    loss = torch.sum(-teacher_out_ref[iq][:,v] * F.log_softmax(student_out_ref[v][:, iq], dim=-1), dim=-1)  # [B, k]
                    ref_loss += loss.mean()
                    ref_n_loss_terms += 1
        global_loss /= global_n_loss_terms
        ref_loss /= ref_n_loss_terms
        total_loss = 0.5 * global_loss + 0.5 * ref_loss

        self.update_center(teacher_output)
        return total_loss, global_loss, ref_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # batch_center: [1, 1+(2+x)*k, D]

        # ema update
        self.center = self.center * self.center_momentum + batch_center[:, 0] * (1 - self.center_momentum)
        if self.another_center:
            self.ref_center = self.ref_center * self.center_momentum + batch_center[:, 1:].mean(dim=1, keepdim=False) * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, num_reference, sampling_mode, min_intersection=0.01):
        self.num_reference = num_reference
        self.sampling_mode = sampling_mode

        # flip = transforms.RandomHorizontalFlip(p=0.5)
        color_jitter = transforms.Compose([    
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.global_crop1 = transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC)
        # second global crop
        self.global_transfo2 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        self.global_crop2 = transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC)
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])
        self.local_crop = transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=InterpolationMode.BICUBIC)

        self.crop_list = [self.global_crop1, self.global_crop2]
        self.crop_list.extend([self.local_crop] * self.local_crops_number)

        # params for leopart locating bbox
        self.min_intersection = min_intersection
        self.nmb_crops = [2, self.local_crops_number]
        self.size_crops = [224, 96]

    # def __call__(self, image):
    #     crops = []
    #     crops.append(self.global_transfo1(image))
    #     crops.append(self.global_transfo2(image))
    #     for _ in range(self.local_crops_number):
    #         crops.append(self.local_transfo(image))
    #     return crops

    def __call__(self, sample: torch.Tensor):
        '''
        function from Leopart repo on github.
        Note for gc_bboxes and otc_bboxes. box param: [x1, y1, x2, y2]. x1y1: left top, x2y2: right bottom.
        gc_bboxes: [2, 2+x, 4], otc_bboxes: [2+x, 2, 4]
        gc_bboxes[i][j]: The intersection region of i-th global crop and j-th all crop, coordinate of i-th global crop image.
        otc_bboxes[i][j]: The intersection regions of i-th all crop and j-th global crop, coordinate of i-th all crop image.
        Key point: The bounding box param is calculate on i-th image.
        Example:
            gc_bboxes[0][1] is exactly the same as otc_bboxes[0][1].
        '''
        multi_crops = []
        crop_bboxes = torch.zeros(len(self.crop_list), 4)
        flip_record = []

        for i, rrc_transform in enumerate(self.crop_list):
            # Get random crop params
            y1, x1, h, w = rrc_transform.get_params(sample, rrc_transform.scale, rrc_transform.ratio)
            if i > 0:
                # Check whether crop has min overlap with existing global crops. If not resample.
                while True:
                    # Calculate intersection between sampled crop and all sampled global crops
                    bbox = torch.Tensor([x1, y1, x1 + w, y1 + h])
                    left_top = torch.max(bbox.unsqueeze(0)[:, None, :2],
                                         crop_bboxes[:min(i, self.nmb_crops[0]), :2])
                    right_bottom = torch.min(bbox.unsqueeze(0)[:, None, 2:],
                                             crop_bboxes[:min(i, self.nmb_crops[0]), 2:])
                    wh = _upcast(right_bottom - left_top).clamp(min=0)
                    inter = wh[:, :, 0] * wh[:, :, 1]

                    # set min intersection to at least 1% of image area
                    min_intersection = int((sample.size[0] * sample.size[1]) * self.min_intersection)
                    # Global crops should have twice the min_intersection with each other
                    if i in list(range(self.nmb_crops[0])):
                        min_intersection *= 2
                    if not torch.all(inter > min_intersection):
                        y1, x1, h, w = rrc_transform.get_params(sample, rrc_transform.scale, rrc_transform.ratio)
                    else:
                        break

            # Apply rrc params and store absolute crop bounding box
            img = resized_crop(sample, y1, x1, h, w, rrc_transform.size, rrc_transform.interpolation)
            crop_bboxes[i] = torch.Tensor([x1, y1, x1 + w, y1 + h])

            # Apply flip
            flip_flag = torch.rand(1) < 0.5
            flip_record.append(flip_flag)
            if flip_flag:
                img = hflip(img)
                # crop_bboxes[i][0] = rrc_transform.size[1] - crop_bboxes[i][2]

            if i == 0:
                img = self.global_transfo1(img)
            elif i == 1:
                img = self.global_transfo2(img)
            else:
                img = self.local_transfo(img)

            multi_crops.append(img)


        # Calculate relative bboxes for each crop pair from aboslute bboxes
        gc_bboxes, otc_bboxes = self.calculate_bboxes(crop_bboxes)

        ref_pos_gc, ref_pos_otc = self.sampling_reference_point(gc_bboxes, otc_bboxes)

        tea_pos = ref_pos_gc.chunk(2)   # (2, [1, 2+x, k*k, 2])
        stu_pos = ref_pos_otc.chunk(ref_pos_otc.shape[0])   # (2+x, [1, 2, k*k, 2])

        # Adjust reference point coordinate according to flip
        for i in range(len(flip_record)):
            if flip_record[i]:
                if i < 2:
                    tea_pos[i][:,:,:,0] = -tea_pos[i][:,:,:,0]
                    stu_pos[i][:,:,:,0] = -stu_pos[i][:,:,:,0]
                else:
                    stu_pos[i][:,:,:,0] = -stu_pos[i][:,:,:,0]          

        tea_pos = list(map(lambda x: x.squeeze(0).detach(), tea_pos))
        stu_pos = list(map(lambda x: x.squeeze(0).detach(), stu_pos))

        # --May not matter--
        # flip_record contains flip_flag, which is a tensor
        # Memory leak occurs without del
        del flip_record
        del gc_bboxes, otc_bboxes

        # multi_crops: List[2+x Tensor[3,H,W]]
        # tea_pos: List[2 Tensor[2+x,k*k,2]]
        # stu_pos: List[2+x Tensor[2,k*k,2]]
        return multi_crops, tea_pos, stu_pos

    def calculate_bboxes(self, crop_bboxes):
        # 1. Calculate two intersection bboxes for each global crop - other crop pair
        gc_bboxes = crop_bboxes[:self.nmb_crops[0]]
        left_top = torch.max(gc_bboxes[:, None, :2], crop_bboxes[:, :2])  # [nmb_crops[0], sum(nmb_crops), 2]
        right_bottom = torch.min(gc_bboxes[:, None, 2:], crop_bboxes[:, 2:])  # [nmb_crops[0], sum(nmb_crops), 2]
        # Testing for non-intersecting crops. This should always be true, just as safe-guard.
        assert torch.all((right_bottom - left_top) > 0)

        # 2. Scale intersection bbox with crop size
        # Extract height and width of all crop bounding boxes. Each row contains h and w of a crop.
        ws_hs = torch.stack((crop_bboxes[:, 2] - crop_bboxes[:, 0], crop_bboxes[:, 3] - crop_bboxes[:, 1])).T[:, None]

        # Stack global crop sizes for each bbox dimension
        crops_sizes = torch.repeat_interleave(torch.Tensor([self.size_crops[0]]), self.nmb_crops[0] * 2)\
            .reshape(self.nmb_crops[0], 2)
        if len(self.size_crops) == 2:
            lc_crops_sizes = torch.repeat_interleave(torch.Tensor([self.size_crops[1]]), self.nmb_crops[1] * 2)\
                .reshape(self.nmb_crops[1], 2)
            crops_sizes = torch.cat((crops_sizes, lc_crops_sizes))[:, None]  # [sum(nmb_crops), 1, 2]

        # Calculate x1s and y1s of each crop bbox
        x1s_y1s = crop_bboxes[:, None, :2]

        # Scale top left and right bottom points by percentage of width and height covered
        left_top_scaled_gc = crops_sizes[:2] * ((left_top - x1s_y1s[:2]) / ws_hs[:2])
        right_bottom_scaled_gc = crops_sizes[:2] * ((right_bottom - x1s_y1s[:2]) / ws_hs[:2])
        left_top_otc_points_per_gc = torch.stack([left_top[i] for i in range(self.nmb_crops[0])], dim=1)
        right_bottom_otc_points_per_gc = torch.stack([right_bottom[i] for i in range(self.nmb_crops[0])], dim=1)
        left_top_scaled_otc = crops_sizes * ((left_top_otc_points_per_gc - x1s_y1s) / ws_hs)
        right_bottom_scaled_otc = crops_sizes * ((right_bottom_otc_points_per_gc - x1s_y1s) / ws_hs)

        # 3. Construct bboxes in x1, y1, x2, y2 format from left top and right bottom points
        gc_bboxes = torch.cat((left_top_scaled_gc, right_bottom_scaled_gc), dim=2)
        otc_bboxes = torch.cat((left_top_scaled_otc, right_bottom_scaled_otc), dim=2)
        return gc_bboxes, otc_bboxes
    
    def sampling_reference_point(self, gc_bboxes, otc_bboxes):
        # gc_bboxes: [2, 2+x, 4], otc_bboxes: [2+x, 2, 4]
        if self.sampling_mode == 'random':
            num_reference_point = self.num_reference * self.num_reference   # To align with "grid" mode
            ref_relate_pos = torch.rand((gc_bboxes.shape[0], gc_bboxes.shape[1], num_reference_point, 2))    # [2, 2+x, k*k, 2]
        elif self.sampling_mode == 'grid':
            raise RuntimeError("Not implemented yet.")
        else:
            # sanity check
            raise RuntimeError("Invalid sampling mode.")

        ref_pos_x_gc = (gc_bboxes[:,:,None,2] - gc_bboxes[:,:,None,0]) * ref_relate_pos[:,:,:,0] + gc_bboxes[:,:,None,0]
        ref_pos_y_gc = (gc_bboxes[:,:,None,3] - gc_bboxes[:,:,None,1]) * ref_relate_pos[:,:,:,1] + gc_bboxes[:,:,None,1]
        ref_pos_gc = torch.stack((ref_pos_x_gc, ref_pos_y_gc), dim=-1)  # [2, 2+x, k*k, 2]
        ref_pos_gc = ref_pos_gc / self.size_crops[0] * 2. - 1.

        ref_relate_pos = ref_relate_pos.permute(1,0,2,3)
        ref_pos_x_all = (otc_bboxes[:,:,None,2] - otc_bboxes[:,:,None,0]) * ref_relate_pos[:,:,:,0] + otc_bboxes[:,:,None,0]
        ref_pos_y_all = (otc_bboxes[:,:,None,3] - otc_bboxes[:,:,None,1]) * ref_relate_pos[:,:,:,1] + otc_bboxes[:,:,None,1]
        ref_pos_all = torch.stack((ref_pos_x_all, ref_pos_y_all), dim=-1)
        size_t = torch.cat((torch.tensor([self.size_crops[0], self.size_crops[0]]), torch.tensor([self.size_crops[1] for i in range(self.local_crops_number)])))
        ref_pos_all = ref_pos_all / size_t[:,None,None,None] * 2. - 1.

        # coordinate in x,y format
        # ref_pos_gc: [2, 2+x, k*k, 2], ref_pos_all: [2+x, 2, k*k, 2]
        return ref_pos_gc, ref_pos_all

def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
