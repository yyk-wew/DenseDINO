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
import os
import sys
import argparse
import cv2
import random
import colorsys
# import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vits

import math



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize cls_token embdding')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    # parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    # parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    # parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument('--num_cls_token', default=1, type=int, help="Number of cls_token")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, num_cls_token=args.num_cls_token)
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
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")

    patch_pos_embed = model.pos_embed[:, args.num_cls_token:].squeeze(0)
    cls_pos_embed = model.pos_embed[:, :args.num_cls_token].squeeze(0)
    N = model.pos_embed.shape[1] - model.num_cls_token
    query_embed = patch_pos_embed[107].unsqueeze(0)
    # query_embed = cls_pos_embed[0]

    # norm
    patch_pos_embed = torch.nn.functional.normalize(patch_pos_embed, p=2, dim=-1)
    cls_pos_embed = torch.nn.functional.normalize(cls_pos_embed, p=2, dim=-1)
    query_embed = torch.nn.functional.normalize(query_embed, p=2, dim=-1)

    # similarity
    sim_mat = patch_pos_embed @ query_embed.T
    sim_mat = sim_mat.reshape(int(math.sqrt(N)), int(math.sqrt(N)))
    print(sim_mat.shape)
    vis_data = sim_mat.cpu().numpy()

    # plot show
    plt.figure(figsize = (10,10))
    plt.imshow(vis_data, interpolation='nearest')
    plt.savefig('embed.png')