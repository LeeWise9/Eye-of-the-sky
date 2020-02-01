# Semantic-Segmentation-of-Satellite-Image
In this project, SegNet, U-Net etc. are used for semantic segmentation of satellite images. Based on keras and runs on GPU.

这是一个应用深度学习方法解决工业问题的应用项目：使用改进的 SegNet、U-net 对卫星图像做语义分割。代码基于Keras编写，支持GPU加速。

## 问题描述

本项目意在通过训练，使神经网络能够完成对高清卫星图像的语义分割。

<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E5%A4%A9%E7%A9%BA%E4%B9%8B%E7%9C%BC1.jpg" alt="Sample"  width="500">
</p>


## 网络结构

本项目主要使用了 SegNet 和 U-Net，这两种网络都是语义分割领域常用的模型。

### SegNet
<p align="center">
	<img src="https://images2017.cnblogs.com/blog/1093303/201801/1093303-20180122200010084-939706515.png" alt="Sample"  width="500">
</p>

## U-Net

<p align="center">
	<img src="https://images2017.cnblogs.com/blog/1093303/201801/1093303-20180122200158397-1275935789.png" alt="Sample"  width="500">
</p>


## 数据集
本项目数据集来自于[DataFountain](https://www.datafountain.cn/#/competitions/270/data-intro)的一次公开赛，由于赛事已经结束，该数据集的官方获取渠道已经关闭。有参赛团队将下载的数据集公开了：[下载地址](https://pan.baidu.com/s/1i6oMukH)，提取密码：yqj2。

需要注意的是，源数据仅包含 5 张高清卫星图和其对应的语义分割 mask 图像，其中 mask 图像为 16 位图像，一般的图像查看器显示为全黑色，需要转为 8 位才能正常查看。
