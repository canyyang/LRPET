# Compact Model Training by Low-Rank Projection with Energy Transfer(TNNLS 2024)

[![arXiv](https://img.shields.io/badge/arxiv-2204.05566-b31b1b?style=plastic&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2311.17132)](https://arxiv.org/abs/2204.05566)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is the official PyTorch implementation for vision transformer of [Compact Model Training by Low-Rank Projection with Energy Transfer](https://arxiv.org/abs/2204.05566). Due to issues with
computing resources and training cycles, we trained the Vision Transformer from scratch on small-size datasets.

## Result
| Method  | FLOPs(PR) | Params(PR) | CIFAR10 | CIFAR100 |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| Basline  | 174.25M(0.0%)  | 2.69M(0.0%)  | 93.54  | 72.49  |
| EViT  | 128.78M(26.1%)  | 2.69M(0.0%) |  92.67  | 71.83  |
| LRPET  | 126.42M(27.4%) | 1.97M(26.8%)  |  92.96  | 72.08  |

## Usage
### Baseline
```
CUDA_VISIBLE_DEVICES=0 nohup python baseline.py > save_log/baseline.log 2>&1 &
```

### LRPET
```
CUDA_VISIBLE_DEVICES=0 nohup python lrpet.py > save_log/lrpet.log 2>&1 &
```

## Reference
https://github.com/youweiliang/evit

https://github.com/aanna0701/SPT_LSA_ViT