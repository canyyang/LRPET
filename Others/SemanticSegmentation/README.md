# Compact Model Training by Low-Rank Projection with Energy Transfer(TNNLS 2024)

[![arXiv](https://img.shields.io/badge/arxiv-2204.05566-b31b1b?style=plastic&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2311.17132)](https://arxiv.org/abs/2204.05566)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is the official PyTorch implementation for semantic segmentation of [Compact Model Training by Low-Rank Projection with Energy Transfer](https://arxiv.org/abs/2204.05566). We we apply LRPET to to the PSPNet. We train the model on PASCAL VOC 2012 and Semantic Boundaries training datasets, and evaluate the performance with mean Intersection over Union on the PASCAL VOC 2012 test dataset.

## Result
| Method  | FLOPs Reduction    | mIOU |
|:--------:|:--------:|:--------:|
| Basline  | 0%  | 76.48  |
| LRPET  | 30%  | 76.33  | 

## Usage
### Baseline
```
CUDA_VISIBLE_DEVICES=0 nohup python baseline.py > save_log/baseline.log 2>&1 &
```

### LRPET
```
CUDA_VISIBLE_DEVICES=0 nohup python lrpet.py > save_log/lrpet.log 2>&1 &
```

### Evaluation
```
CUDA_VISIBLE_DEVICES=0 nohup python get_moiu.py > save_log/miou.log 2>&1 &
```

## Reference
https://github.com/bubbliiiing/pspnet-pytorch