# Compact Model Training by Low-Rank Projection with Energy Transfer(TNNLS 2024)

[![arXiv](https://img.shields.io/badge/arxiv-2204.05566-b31b1b?style=plastic&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2311.17132)](https://arxiv.org/abs/2204.05566)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is the official PyTorch implementation for convolution network compression on CIFAR10 dataset of [Compact Model Training by Low-Rank Projection with Energy Transfer](https://arxiv.org/abs/2204.05566).

## Usage
Use python cifar10_train.py to train a new model. Here is some example settings:
```
CUDA_VISIBLE_DEVICES=0 nohup python cifar10_train.py  --model resnet56 --prun_goal 0.70 --redu_fac 0 --epochs 400 >SVD/CNN/cifar10/save_log/resnet56_prun0.80.log 2>&1 &
```

Use python search.py to search a model. Here is a example setting:
```
CUDA_VISIBLE_DEVICES=0 nohup python search.py  --model resnet56  --redu_fac 0 --epochs 60 >cifar10/save_log/resnet56_prun0.7_search.log 2>&1 &
```
## Result
```
CUDA_VISIBLE_DEVICES=0 nohup python imagenet_resnet_trans_train.py  -a resnet34 --prun_goal 0.58 > imagenet/save_log/imagenet_resnet34_prun0.58.log 2>&1 &
```
```
CUDA_VISIBLE_DEVICES=0 nohup python imagenet_resnet_trans_train.py  -a resnet50 --prun_goal 0.62 > imagenet/save_log/imagenet_resnet50_prun0.62.log 2>&1 &
```
