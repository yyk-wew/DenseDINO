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
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torchvision.transforms.functional import InterpolationMode

import utils
import vision_transformer as vits

from segment_utils import VOCDataset, PredsmIoU, StreamSegMetrics
from tqdm import tqdm


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    if 'leopart' in args.pretrained_weights:
        state_dict = torch.load(args.pretrained_weights)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        state_dict['patch_pos_embed'] = state_dict['pos_embed'][:, 1:, :]
        state_dict['cls_pos_embed'] = state_dict['pos_embed'][:, :1, :]
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    if args.head_type == 'linear':
        seg_head = nn.Conv2d(embed_dim, args.num_classes, 1)
        nn.init.normal_(seg_head.weight, mean=0, std=0.01)
        nn.init.zeros_(seg_head.bias)
    elif args.head_type == 'fcn':
        pass
    
    seg_head = seg_head.cuda()
    seg_head = nn.parallel.DistributedDataParallel(seg_head, device_ids=[args.gpu])

    # ============ preparing data ... ============
    val_image_transform = pth_transforms.Compose([
        pth_transforms.Resize((448, 448)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_target_transform = pth_transforms.Compose([
        pth_transforms.Resize((448, 448), interpolation=InterpolationMode.NEAREST),
        pth_transforms.ToTensor(),
    ])
    dataset_val = VOCDataset(root=args.data_path, image_set='val', transform=val_image_transform, target_transform=val_target_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    if args.evaluate:
        # utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size)
        assert os.path.isfile(args.pretrained_head)
        if 'leopart' in args.pretrained_head:
            state_dict = torch.load(args.pretrained_head)
            state_dict = {k.replace('finetune_head', 'module'):v for k,v in state_dict.items()}
        else:
            state_dict = torch.load(args.pretrained_head)['state_dict']
        msg = seg_head.load_state_dict(state_dict, strict=True)
        print('Pretrained head found at {} and loaded with msg: {}'.format(args.pretrained_head, msg))
        spatial_size = 448 // args.patch_size
        test_stats = validate_network(val_loader, model, seg_head, args.num_classes, spatial_size, args.eval_size)
        print(f"Accuracy of the network on the {len(dataset_val)} val images: {test_stats['Mean IoU']:.3f}%")
        return

    # train_transform = pth_transforms.Compose([
    #     pth_transforms.RandomResizedCrop(224),
    #     pth_transforms.RandomHorizontalFlip(),
    #     pth_transforms.ToTensor(),
    #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset_train,
    #     sampler=sampler,
    #     batch_size=args.batch_size_per_gpu,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )
    # print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # # set optimizer
    # optimizer = torch.optim.SGD(
    #     linear_classifier.parameters(),
    #     args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
    #     momentum=0.9,
    #     weight_decay=0, # we do not apply weight decay
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # # Optionally resume from a checkpoint
    # to_restore = {"epoch": 0, "best_acc": 0.}
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "checkpoint.pth.tar"),
    #     run_variables=to_restore,
    #     state_dict=linear_classifier,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    # )
    # start_epoch = to_restore["epoch"]
    # best_acc = to_restore["best_acc"]

    # for epoch in range(start_epoch, args.epochs):
    #     train_loader.sampler.set_epoch(epoch)

    #     train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
    #     scheduler.step()

    #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                  'epoch': epoch}
    #     if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
    #         test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
    #         print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #         best_acc = max(best_acc, test_stats["acc1"])
    #         print(f'Max accuracy so far: {best_acc:.2f}%')
    #         log_stats = {**{k: v for k, v in log_stats.items()},
    #                      **{f'test_{k}': v for k, v in test_stats.items()}}
    #     if utils.is_main_process():
    #         with (Path(args.output_dir) / "log.txt").open("a") as f:
    #             f.write(json.dumps(log_stats) + "\n")
    #         save_dict = {
    #             "epoch": epoch + 1,
    #             "state_dict": linear_classifier.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #             "scheduler": scheduler.state_dict(),
    #             "best_acc": best_acc,
    #         }
    #         torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    # print("Training of the supervised linear classifier on frozen features completed.\n"
    #             "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


# def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
#     linear_classifier.train()
#     # metric_logger = 
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     for (inp, target) in metric_logger.log_every(loader, 20, header):
#         # move to gpu
#         inp = inp.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)

#         # forward
#         with torch.no_grad():
#             if "vit" in args.arch:
#                 intermediate_output = model.get_intermediate_layers(inp, n)
#                 output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
#                 if avgpool:
#                     output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
#                     output = output.reshape(output.shape[0], -1)
#             else:
#                 output = model(inp)
#         output = linear_classifier(output)

#         # compute cross entropy loss
#         loss = nn.CrossEntropyLoss()(output, target)

#         # compute the gradients
#         optimizer.zero_grad()
#         loss.backward()

#         # step
#         optimizer.step()

#         # log 
#         torch.cuda.synchronize()
#         metric_logger.update(loss=loss.item())
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, seg_head, num_classes, spatial_size, eval_size):
    seg_head.eval()
    # metric = PredsmIoU(num_pred_classes=num_classes, num_gt_classes=num_classes)
    # metric = metric.cuda()
    metric = StreamSegMetrics(n_classes=num_classes)
    
    with torch.no_grad():
        for i, (inp, target) in enumerate(tqdm(val_loader)):
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)

            batch_size = inp.shape[0]
            # forward            
            output = model.get_intermediate_layers(inp, n=1)[0]
            output = output[:, 1:].reshape(batch_size, spatial_size, spatial_size, model.embed_dim).permute(0, 3, 1, 2)
            output = nn.functional.interpolate(output, size=(eval_size, eval_size), mode='bilinear')
            preds = seg_head(output)
            preds = torch.argmax(preds, dim=1).unsqueeze(1) # [B, 1, 448, 448]

            gt = target * 255   # [B, 1, 448, 448]
            gt = nn.functional.interpolate(gt, size=(eval_size, eval_size), mode='nearest')

            metric.update(label_trues=gt.numpy(), label_preds=preds.cpu().numpy())

            # print(gt, preds)
            # sys.exit(0)

    # calculate miou
    # miou = metric.compute(True, linear_probe=True)[0]
    result = metric.get_results()
    print(result)
    metric.reset()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--pretrained_head', default='', type=str, help="Path to pretrained head weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/pascalvoc/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')

    parser.add_argument("--head_type", type=str, choices=['linear', 'fcn'], default="linear")
    parser.add_argument('--num_classes', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--eval_size', default=448, type=int, help='Size of groundtruth mask')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    eval_linear(args)
