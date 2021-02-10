# Human-Segmentation-with-Dynamic-LiDAR-Data
Sample code for Human Segmentation with Dynamic LiDAR Data.

## Introduction
This is the official code of [Human Segmentation with Dynamic LiDAR Data](https://arxiv.org/abs/2010.08092).

## Requirement
Keras 2.2.4

## Quick start
### Install
git clone https://github.com/tzhong518/Human-Segmentation-with-Dynamic-LiDAR-Data
### Data preparation
Please download [dynamic LiDAR data](https://github.com/Likarian/AutomaticLabeledLiDARSequence)
### Train and test
```
python Train.py \
       --model model's name \
       --gpu 0 \
       --frame 04
```
```
python Test.py \
       --test Test \
       --network /path/to/model \
       --gpu 0 \
       --frame 04
```

## Citation
If you find this work or code is helpful in your research, please cite:
```
Zhong T, Kim W, Tanaka M, et al. Human Segmentation with Dynamic LiDAR Data[J]. arXiv preprint arXiv:2010.08092, 2020.
```
