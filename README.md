# DenseDINO
Pytorch Implementation of paper: "DenseDINO: Boosting Dense Self-Supervised Learning with Token-Based Point-Level Consistency", accepted by IJCAI 2023. [Arxiv Link.](https://arxiv.org/abs/2306.04654)

This repo is based on [DINO](https://github.com/facebookresearch/dino) and pick part of the segmentation evaluation codes from [Leopart](https://github.com/MkuuWaUjinga/leopart).

Please check these repos for environment building and dataset preparation.

## Usage
Training ViT-Small with our methods for 300 epochs on ImageNet:

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_tiny --data_path /path/to/imagenet/ --output_dir /path/to/output/ --batch_size_per_gpu 64 --num_workers 4 --local_crops_number 4 --global_crops_scale 0.25 1. --local_crops_scale 0.25 1. --local_crops_size 224 --saveckp_freq 10 --epochs 100 --given_pos --num_reference 2 --mask_mode all2pos_pos2cls_eye --use_global_loss --use_ref_loss --use_all_crop_ref
```

Eval on ImageNet:
```bash
python -m torch.distributed.launch --nproc_per_node=8 eval_knn.py --arch vit_tiny --pretrained_weights /path/to/ckpt --checkpoint_key teacher --data_path /path/to/imagenet --with_learnable_token
```

Eval on PascalVOC:
```bash
python -m torch.distributed.launch --nproc_per_node=8 eval_segment.py --arch vit_tiny --patch_size 16 --data_path /path/to/pascalvoc/ --pretrained_weights /path/to/ckpt/ --output_dir /path/to/log_dir/ --batch_size_per_gpu 8 --num_classes 21 --eval_size 448
```

## Citation
```
@inproceedings{ijcai2023p188,
  title={DenseDINO: Boosting Dense Self-Supervised Learning with Token-Based Point-Level Consistency},
  author={Yuan, Yike and Fu, Xinghe and Yu, Yunlong and Li, Xi},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={1695--1703},
  year={2023},
}
```