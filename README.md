# Human-Segmentation-with-Dynamic-LiDAR-Data
Sample code for Human Segmentation with Dynamic LiDAR Data.

## Introduction
This is the official code of [Human Segmentation with Dynamic LiDAR Data](https://arxiv.org/abs/2010.08092).
![image](https://github.com/tzhong518/Human-Segmentation-with-Dynamic-LiDAR-Data/blob/main/figures/structure.png)

## Requirement
Keras 2.2.4

## Quick start
### Install
git clone https://github.com/tzhong518/Human-Segmentation-with-Dynamic-LiDAR-Data
### Data preparation
Please download data from [Learning-Based Human Segmentation and Velocity Estimation Using Automatic Labeled LiDAR Sequence for Training](https://github.com/Likarian/AutomaticLabeledLiDARSequence).

You can download weight of 4-frame model from [here](https://drive.google.com/file/d/1w4ZfrfCbxWB7x1gjPsdiD3Y_1Rg-YaeN/view?usp=sharing)
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

