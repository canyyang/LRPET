# Compact Model Training by Low-Rank Projection with Energy Transfer(TNNLS 2024)

[![arXiv](https://img.shields.io/badge/arxiv-2204.05566-b31b1b?style=plastic&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2311.17132)](https://arxiv.org/abs/2204.05566)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is the official PyTorch implementation of [Compact Model Training by Low-Rank Projection with Energy Transfer](https://arxiv.org/abs/2204.05566).

## Introduction
In this paper, we devise a new training method, low-rank projection with energy transfer (LRPET), that trains low-rank compressed networks from scratch and achieves competitive performance.   Comprehensive experiments on image classification, object detection and semantic segmentation have justified that our method is superior to other low-rank compression methods and also outperforms recent state-of-the-art pruning methods.

![](svd.png) 

## Results

### CIFAR-10
| Model  | Flops | Params | Accuracy |
|:--------:|:--------:|:--------:|:--------:|
| ResNet56  | 56.06M  | 0.39M  | 93.27  |
| ResNet110  | 93.78M  | 0.65M |  93.61  |
| VGG16  | 81.29M | 1.62M  |  93.52  |
| GoogleNet  | 0.74B | 3.03M  |  95.25  |
| FBNet  | 218.11M | 2.14M  |  94.66  |
| Ghost-ResNet56  | 41.64M | 0.28M  |  91.63  |

### ImageNet
| Model  | Flops | Params | Top-1 Acc | Top-5 Acc |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| ResNet18  | 0.86B | 5.81M  | 67.87 | 88.04 |
| ResNet34  | 1.71B | 10.51M | 72.95 | 90.98 |
| ResNet50  | 1.90B | 12.89M | 74.25 | 91.93 |

## Usage
### CIFAR-10
Use python cifar10_train.py to train a new model. Here is some example settings:
```
CUDA_VISIBLE_DEVICES=0 nohup python cifar10_train.py  --model resnet56 --prun_goal 0.80 --redu_fac 0 --epochs 400 >SVD/CNN/cifar10/save_log/resnet56_prun0.80.log 2>&1 &
```

Use python search.py to search a model. Here is a example setting:
```
CUDA_VISIBLE_DEVICES=0 nohup python search.py  --model resnet56  --redu_fac 0 --epochs 60 >cifar10/save_log/resnet56_prun0.7_search.log 2>&1 &
```
### ImageNet
```
CUDA_VISIBLE_DEVICES=0 nohup python imagenet_resnet_trans_train.py  -a resnet34 --prun_goal 0.58 > imagenet/save_log/imagenet_resnet34_prun0.58.log 2>&1 &
```

## Notes
We train the networks from scratch with LRPET for 400 epochs on CIFAR10 dataset and 120 epochs on ImageNet dataset. We note that this does not seem fair to the equivalent of partial comparison methods. For this part of the paper, we conducted further experiments to verify the excellence of our method while ensuring the same training cycle.

### CIFAR10
| Model  | Method | FLOPs | Accuracy |
|:--------:|:--------:|:--------:|:--------:|
| ResNet56  | 174.2<br>5M(0.0%)  | 2.69M(0.0%)  | 93.54  |
| EViT  | 128.78M(26.1%)  | 2.69M(0.0%) |  92.67  |
| LRPET  | 126.42M(27.4%) | 1.97M(26.8%)  |  92.96  |

## Others
We also apply LRPET in object detection, semantic segmentation and vision transformer models. The specific experimental code and results are also attached.

## Citation
```
@ARTICLE{10551437,
  author={Guo, Kailing and Lin, Zhenquan and Chen, Canyang and Xing, Xiaofen and Liu, Fang and Xu, Xiangmin},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Compact Model Training by Low-Rank Projection With Energy Transfer}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Training;Matrix decomposition;Energy exchange;Sparse matrices;Tensors;Image coding;Convolution;Energy transfer;low-rank projection (LRP);network compression;training method},
  doi={10.1109/TNNLS.2024.3400928}}
```

