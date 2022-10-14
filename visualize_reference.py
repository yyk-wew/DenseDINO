import argparse
import os
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
from torchvision.transforms.functional import resized_crop

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

from main_dino import DataAugmentationDINO

import matplotlib.pyplot as plt
import numpy as np

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
    
    parser.add_argument('--num_cls_token', default=1, type=int,
        help="Number of cls_token")
    parser.add_argument('--given_pos', action='store_true', help='Replace cls_pos_embed with interpolated patch_pos_embed.')

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


def vis_pos(x, pos, model, images, b_index, crop_index, target_index, name):
    # -- model.prepare_token --
    B, nc, w, h = x.shape
    x = model.patch_embed(x)  # patch linear embedding
    
    # add the [CLS] token to the embed patch tokens
    if model.given_pos:
        num_cls_token = pos.shape[1] * pos.shape[2] + 1
        if model.with_cls_token:   
            cls_tokens = model.cls_token.expand(1, num_cls_token, -1)
        else:
            cls_tokens = torch.zeros((1, num_cls_token - 1, model.embed_dim)).to(model.cls_token.device)
            cls_tokens = torch.cat((model.cls_token, cls_tokens), dim=1)
    else:
        num_cls_token = 1
        cls_tokens = model.cls_token
    cls_tokens = cls_tokens.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)   # [B, 1+x+N, C]

    patch_pos = model.interpolate_pos_encoding(x, w, h, num_cls_token)

    if model.given_pos:
        cls_pos_embed = model.interpolate_ref_point_pos_encoding(x, pos, patch_pos)
        pos_embed = torch.cat((model.cls_pos_embed.expand(B, -1, -1), cls_pos_embed, patch_pos.expand(B, -1, -1)), dim=1)
    else:
        pos_embed = torch.cat((model.cls_pos_embed, patch_pos), dim=1).expand(B, -1, -1)

    # -- dino loss --
    cls_pos_embed = cls_pos_embed.unflatten(1, (2, cls_pos_embed.shape[1] // 2))
    print("cls_pos_embed", cls_pos_embed.shape)

    # images: 2 * [2B,C,H,W]
    # x: [2B, 1+2*k+N, embed]
    # patch_pos: [1, 196, embed]
    # cls_pos_embed: [2B, 2, k, embed]
    
    actual_B = B // 2   # Attention: only fits when using crop 2, local crop

    patch_pos = patch_pos.squeeze(0)    # [196, embed]
    cls_pos_embed = cls_pos_embed[crop_index * actual_B + b_index][target_index]    # [k, embed]
    patch_pos = torch.nn.functional.normalize(patch_pos, p=2, dim=-1)
    cls_pos_embed = torch.nn.functional.normalize(cls_pos_embed, p=2, dim=-1)

    sim_mat = patch_pos @ cls_pos_embed.T
    sim_mat = sim_mat.reshape(int(math.sqrt(patch_pos.shape[0])), int(math.sqrt(patch_pos.shape[0])), -1)
    print("sim_mat", sim_mat.shape)    # should be [14, 14, k]
    vis_data = sim_mat.cpu().numpy()

    img = images[crop_index][b_index]    # [3, H, W]
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
        )
    img = inv_normalize(img.unsqueeze(0))
    img = img.squeeze(0).permute(1,2,0).cpu().numpy()  # [H, W, 3]
    print(img.shape)

    pos = pos[crop_index * actual_B + b_index][target_index]  # [k, 2]
    pos = pos.cpu().numpy()

    pos = (pos + 1.) / 2. * 224.

    num_plots = pos.shape[-2] + 1
    fig, ax = plt.subplots(1, num_plots,figsize=((num_plots)*5, 5))
    color_list = ['red', 'blue', 'yellow', 'green']
    for j in range(num_plots):
        if j == 0:
            ax[j].imshow(img)
            for k in range(pos.shape[-2]):
                ax[j].scatter(pos[k][0], pos[k][1], marker='o', c=color_list[k])
        else:
            ax[j].imshow(vis_data[:,:,j-1], interpolation='nearest')
    
    fig.savefig(name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Visualize Reference points.')

    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')

    parser.add_argument('--given_pos', action='store_true', help='Replace cls_pos_embed with interpolated patch_pos_embed.')
    parser.add_argument('--with_cls_token', action='store_true', help='Reference token shared learnable class token.')
    parser.add_argument('--another_center', action='store_true', help='Use separate centering for given_pos_token.')
    parser.add_argument('--num_reference', default=1, type=int, help="Number of points sampled per crop. Use k*k points in actual.")
    parser.add_argument('--sampling_mode', type=str, choices=['random', 'grid'], help='Mode of reference point sampling.')

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

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, given_pos=args.given_pos, with_cls_token=args.with_cls_token)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # adapt to patch_pos_embed
        print(state_dict['pos_embed'].shape)
        state_dict['patch_pos_embed'] = state_dict['pos_embed'][:, 1:, :]
        state_dict['cls_pos_embed'] = state_dict['pos_embed'][:, :1, :]
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    transform = DataAugmentationDINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
            args.num_reference,
            args.sampling_mode
        )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    for it, ((images, tea_pos, stu_pos), _) in enumerate(data_loader):
        
        images = [im.cuda(non_blocking=True) for im in images]
        tea_pos = [pos.cuda(non_blocking=True) for pos in tea_pos]
        stu_pos = [pos.cuda(non_blocking=True) for pos in stu_pos]

        
        # -- Multi-crop Wrapper --
        start_idx, end_idx = 0, 2
        x = torch.cat(images[start_idx: end_idx])
        pos_tea = torch.cat(tea_pos[start_idx:end_idx])
        pos_stu = torch.cat(stu_pos[start_idx:end_idx])
        
        vis_pos(x, pos_tea, model, images, b_index=0, crop_index=0, target_index=1, name='tea_vis.png')
        vis_pos(x, pos_stu, model, images, b_index=0, crop_index=1, target_index=0, name='stu_vis.png')

        break
        