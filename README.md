# Semantic-Segmentation-of-Satellite-Image
In this project, SegNet, U-Net etc. are used for semantic segmentation of satellite images. Based on keras and runs on GPU.

# 基于 Segnet 和 U-Net 的遥感图像语义分割

本项目中将使用改进的 SegNet 和 U-Net 对卫星图像做语义分割，基于 Keras 编写，支持 GPU 加速。

## SegNet 网络结构

<p align="center">
	<img src="https://images2017.cnblogs.com/blog/1093303/201801/1093303-20180122200010084-939706515.png" alt="Sample"  width="500">
</p>


## U-Net 网络结构

<p align="center">
	<img src="https://images2017.cnblogs.com/blog/1093303/201801/1093303-20180122200158397-1275935789.png" alt="Sample"  width="500">
</p>


## 数据集

[下载地址](https://pan.baidu.com/s/1i6oMukH)，提取密码：yqj2。

## 注意

* 1.数据 label 是深度为 16 的图像，无法使用 8 位图片查看器显示。

* 2.推理时的图片通常为大尺寸高清图，需要先分割再输入到神经网络，将输出结果再拼接。

* 3.上述分割后再拼接的思路易导致拼接边缘不平滑，可对输出结果做更进一步的剪裁与拼接。

* 4.多模型融合采取投票的策略可以进一步改进，提升效果。
