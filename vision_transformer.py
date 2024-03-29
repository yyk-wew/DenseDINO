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
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # -- Masked Attention for reference tokens --
        if attn_mask is not None:
            attn = attn + attn_mask
        # -------------------------------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_learnable_token=False,
                 remove_global_token=False, detach_pos_embed=False, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.with_learnable_token = with_learnable_token
        self.remove_global_token = remove_global_token
        self.detach_pos_embed = detach_pos_embed

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if not self.remove_global_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos_embed, std=.02)
        if self.with_learnable_token:
            self.ref_learn_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.ref_learn_token, std=.02)
        
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        trunc_normal_(self.patch_pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1]
        N = self.patch_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.patch_pos_embed
        
        patch_pos_embed = self.patch_pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def interpolate_ref_point_pos_encoding(self, pos, patch_pos_embed):
        # patch_pos_embed: [1, N, embed]
        # pos: [B, x, k, 2]
        # return: [B, x, k, embed]

        N = patch_pos_embed.shape[1]
        dim = patch_pos_embed.shape[-1]
        B = pos.shape[0]
        
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2).expand(B, -1, -1, -1)
        # patch_pos_embed: [B, embed, H, W]

        if self.detach_pos_embed:
            patch_pos_embed = patch_pos_embed.detach()
            
        ref_pos_embed = nn.functional.grid_sample(input=patch_pos_embed, grid=pos, mode='bicubic')  # [B, embed, x, k]
        ref_pos_embed = ref_pos_embed.flatten(2,3).permute(0, 2, 1)   # [B, x*k, embed]

        return ref_pos_embed

    def prepare_tokens(self, x, pos=None, mask_mode=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding: [B, N, D]

        # interpolate patch positional encoding
        patch_pos = self.interpolate_pos_encoding(x, w, h)  # [1, N, D]

        # add the [REF] token to the embed patch tokens
        num_ref_token = 0
        if pos is not None:
            num_ref_token = pos.shape[1] * pos.shape[2]
            if self.with_learnable_token:   
                ref_tokens = self.ref_learn_token.expand(1, num_ref_token, -1)
            else:
                ref_tokens = torch.zeros((1, num_ref_token, self.embed_dim)).to(x.device)

            ref_tokens = ref_tokens.expand(B, -1, -1)
            x = torch.cat((ref_tokens, x), dim=1)   # [B, x+N, D]

        # add the [CLS] token to the embed patch tokens
        num_cls_token = 0
        if not self.remove_global_token:
            num_cls_token = 1
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)   # [B, 1+x+N, D]
        
        pos_embed = patch_pos.expand(B, -1, -1)
        # add REF positional encoding
        if pos is not None:
            ref_pos_embed = self.interpolate_ref_point_pos_encoding(pos, self.patch_pos_embed) # [B, x, D]
            pos_embed = torch.cat((ref_pos_embed, pos_embed), dim=1)
        
        # add CLS positional encoding
        if not self.remove_global_token:
            pos_embed = torch.cat((self.cls_pos_embed.expand(B, -1, -1), pos_embed), dim=1)

        # add positional encoding to each token
        x = x + pos_embed

        # prepare attn mask
        attn_mask = self.prepare_attn_mask(x, mask_mode, num_cls_token, num_ref_token)

        return self.pos_drop(x), attn_mask

    def forward(self, x, pos=None, mask_mode=None, abl=False):
        if abl:
            return self.forward_ablation(x, pos)
        else:
            return self.forward_impl(x, pos, mask_mode)

    def forward_impl(self, x, pos, mask_mode):
        x, attn_mask = self.prepare_tokens(x, pos, mask_mode)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm(x)
        
        num_cls_token = 0 if self.remove_global_token else 1
        cls_token = x[:, :num_cls_token]
        
        if self.training:
            num_ref_token = pos.shape[1] * pos.shape[2] if pos is not None else 0
            ref_token = x[:, num_cls_token:num_cls_token+num_ref_token]
            return cls_token, ref_token
        else:
            return cls_token.squeeze(1)

    def forward_ablation(self, x, pos):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding: [B, N, D]

        # interpolate patch positional encoding
        patch_pos = self.interpolate_pos_encoding(x, w, h)  # [1, N, D]

        # add the [CLS] token to the embed patch tokens
        num_cls_token = 0
        if not self.remove_global_token:
            num_cls_token = 1
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)   # [B, 1+x+N, D]
        
        pos_embed = patch_pos.expand(B, -1, -1)
        
        # add CLS positional encoding
        if not self.remove_global_token:
            pos_embed = torch.cat((self.cls_pos_embed.expand(B, -1, -1), pos_embed), dim=1)

        # add positional encoding to each token
        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        cls_token = x[:, :num_cls_token]
        
        patch_tokens = x[:, num_cls_token:]  

        N = patch_tokens.shape[1]
        dim = patch_tokens.shape[-1]
        B = pos.shape[0]
        
        patch_tokens = patch_tokens.reshape(B, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        # patch_tokens: [B, embed, H, W]
            
        ref_token = nn.functional.grid_sample(input=patch_tokens, grid=pos, mode='bicubic')  # [B, embed, x, k]
        ref_token = ref_token.flatten(2,3).permute(0, 2, 1)   # [B, x*k, embed]

        return cls_token, ref_token


    def prepare_attn_mask(self, x, mask_mode, num_cls_token=0, num_ref_token=0):
        N = x.shape[1]

        mask = torch.zeros((N, N)).to(x.device)

        if mask_mode == '020':
            return None
        elif mask_mode == 'all2pos':
            mask[:, num_cls_token:num_cls_token+num_ref_token] = -float('inf')
        elif mask_mode == 'all2pos_pos2cls':
            mask[:, num_cls_token:num_cls_token+num_ref_token] = -float('inf')
            mask[num_cls_token:num_cls_token+num_ref_token, 0] = -float('inf')
        elif mask_mode == 'all2pos_pos2cls_eye':
            mask[:, num_cls_token:num_cls_token+num_ref_token] = -float('inf')
            mask[num_cls_token:num_cls_token+num_ref_token, 0] = -float('inf')
            mask = mask.fill_diagonal_(0.)
        elif mask_mode == "all_query":
            mask[:, : num_cls_token+num_ref_token] = -float('inf')
            mask = mask.fill_diagonal_(0.)
        elif mask_mode is None:
            return None
        else:
            print("Warning: Invalid mask mode: ", mask_mode)
            return None
        
        return mask.detach()
      

    def get_last_selfattention(self, x, pos=None, mask_mode=None, n=1):
        x, attn_mask = self.prepare_tokens(x, pos, mask_mode)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - n:
                x = blk(x, attn_mask=attn_mask)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, pos=None, mask_mode=None, n=1):
        x, attn_mask = self.prepare_tokens(x, pos, mask_mode)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_patch_tokens(self, x, pos=None, mask_mode=None):
        x, attn_mask = self.prepare_tokens(x, pos, mask_mode)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm(x)

        num_other_token = 0 if self.remove_global_token else 1
        num_other_token = num_other_token + pos.shape[1] * pos.shape[2] if pos is not None else num_other_token
        x = x[:, num_other_token:]  
        
        return x


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
