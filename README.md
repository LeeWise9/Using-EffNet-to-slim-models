# Using EffNet to slim models
This project will explore how the efficiency network can improve the calculation efficiency and reduce the size of the model. Taking the steering wheel angle prediction network as an example, it runs on the GPU based on keras.


本项目将探讨[EffNet](https://arxiv.org/abs/1801.06434)是如何提高计算效率并减小模型大小的，该项目以方向盘角度预测网络为例，基于keras，在gpu上运行。


本项目主要包含以下几个部分：<br>
* 1.空间可分离卷积（Spatial Separable Convolution）；<br>
* 2.深度可分离卷积（Depthwise Separable Convolution）；<br>
* 3.EffNet和MobileNets
* 4.将EffNet应用到手写数字识别；<br>
* 5.将EffNet应用到方向盘角度预测。<br>


## 1.空间可分离卷积（Spatial Separable Convolution）<br>
Spatial Separable Convolution 将普通卷集核拆分为两个更小的卷积核，例如，最常用的情况是将 3x3 的卷积核拆分为 1x3 和 3x1 的卷积核。

以3x3卷积核为例,输出一个通道,普通卷积需要的参数为32xDxDxM，其中D为输入特征图像的长/宽，M为输入图像的通道数。

若使用空间可分离卷积，先使用1x3的卷积核后使用3x1的卷积核,计算量为(3+3)xDxDxM，参数量为普通卷积的6/9。

推广到一般情况，以nxn(n>2)卷积核为例，计算量从 nxnxDxM 减少到了 (n+n)xDxM。

空间可分离卷积示意图如下：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%A9%BA%E9%97%B4%E5%8F%AF%E5%88%86%E7%A6%BB%E5%8D%B7%E7%A7%AF.png" alt="Sample"  width="500">
</p>


## 2.深度可分离卷积（Depthwise Separable Convolution）<br>
Depthwise Separable Convolution 将普通卷积分为两个步骤，下面举例说明。

假设原始图像大小是12x12，有三个通道RGB，其输入图片格式是：12x12x3。滤波器窗口大小是5x5x3，得到的输出图像大小是8x8x1（padding模式是valid）。<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%99%AE%E9%80%9A%E5%8D%B7%E7%A7%AF.jpg" alt="Sample"  width="500">
</p>

一个5x5x3滤波器得到的输出图像8x8x1，设通道数为256，不考虑偏置项，需要：5x5x3x256 = 19200 个参数，计算量为：5x5x3x256x8x8 = 1228800 。如下图所示：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%99%AE%E9%80%9A%E5%8D%B7%E7%A7%AF2.jpg" alt="Sample"  width="500">
</p>

正常卷积的问题在于，卷积核将对图片的所有通道做卷积计算的。那么输出每增加一个通道，卷积核就要增加一个。


深度可分离卷积分为两步：<br>

第一步，对三个通道分别做卷积，输出三个通道的属性：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%B7%B1%E5%BA%A6%E5%8F%AF%E5%88%86%E7%A6%BB1.jpg" alt="Sample"  width="500">
</p>


第二步，用卷积核1x1x3对三个通道再次做卷积，这个时候的输出就和普通卷积一样，是8x8x1：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%B7%B1%E5%BA%A6%E5%8F%AF%E5%88%86%E7%A6%BB2.jpg" alt="Sample"  width="500">
</p>


如果要提取更多的属性，则需要设计更多的1x1x3卷积核心就可以：
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%B7%B1%E5%BA%A6%E5%8F%AF%E5%88%86%E7%A6%BB3.jpg" alt="Sample"  width="500">
</p>

于是深度可分离卷积的参数量为：5x5x1x3+1x1x3x256 = 843，为普通卷积的(1/256)+(1/5x5)=4.39%；计算量为：5x5x3x8x8+256x1x1x3x8x8 = 53952 ，同样是普通卷积的4.39%。

可以推广一下，如果通道数更大、普通卷积的卷积核越大，那么深度可分离卷积就能够节省更多的参数。

## 3.EffNet和MobileNets<br>
为了将神经网络应用到移动或嵌入式平台上，[EffNet](https://arxiv.org/abs/1801.06434)和[MobileNets](https://arxiv.org/abs/1704.04861)被提出。MobileNets基于流线型架构，使用深度可分离卷积来构建轻量级深度神经网络。EffNet对MobileNets的轻量级的模型进行了进一步优化，同时使用深度可分离卷积与空间可分离卷积，模型进一步减小。
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/effnet%2Cmobilenets.jpg" alt="Sample"  width="500">
</p>

MobileNets中已经使用了深度可分离卷积，EffNet在此基础上添加了空间可分离卷积，进一步减少了参数量。

下图是EffNet、MobileNets和ShuffleNet在Cifar10数据集上的测试结果：
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/effnet%2Cmobilenets3.jpg" alt="Sample"  width="500">
</p>

可以发现，在计算量几乎相同的情况下，EffNet比传统神经网络拥有更高的准确率；与MobileNet和ShuffleNet相比同样具有更好的网络性能。


## 4.将EffNet应用到手写数字识别<br>
原文作者提出如下图所示的网络架构：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/effnet%2Cmobilenet4.jpg" alt="Sample"  width="800">
</p>

可以发现Baseline的第一个卷积层为3x3x64 + mp，对应于EffNet的第一组卷积层为 1x1x32 + (dw1x3 + 1d mp) + dw 3x1 + (2x1x64 + 1d stride)。其中mp为池化层，dw为深度可分离卷积（可以通过[keras.layers.DepthwiseConv2D](https://keras.io/zh/layers/convolutional/#depthwiseconv2d)实现）。下面逐步作解释。

1. 1x1x32：使用32个1x1的卷积核做卷积；<br>
2. dw1x3 + 1d mp：使用一个行向量做深度可分离卷积，再加上一个一维池化层，行步长为2，列步长为1；<br>
3. dw 3x1：使用一个列向量做深度可分离卷积；<br>
4. 2x1x64 + 1d stride：使用64个2x1的卷积核做普通卷积，列步长为2，行步长为1。

需要注意的是，Baseline的特征通道数为64，在EffNet中，第1步输入端通道数为32，第4步输出端通道数为64，这样设置保证了两者的参数量相差不太大，使计算结果更具可比性。另外很重要一点是，EffNet在第2步做完dw1x3之后有一个一维池化 1d mp，将图片宽度变为原来的一半，之后在第4步设置列步长为2，使图片长度变为原来的一半，这与Baseline的输出图在通道数和图片尺寸上相同。

将EffNet应用到手写数字识别，可获得如下结果：

Baseline参数量：147658；经20个epochs后：train_acc: 0.9907，val_loss: 0.9943；正确率变化曲线如下：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/base_model.jpg" alt="Sample"  width="500">
</p>

EffNet参数量：68090；经20个epochs后：train_acc: 0.9673，val_loss: 0.9798；经50个epochs后：train_acc: 0.9740，val_loss: 0.9832；正确率变化曲线如下：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/eff_model.jpg" alt="Sample"  width="500">
</p>

EffNet用了Baseline46.11%的参数量，得到了98.88%的性能。


## 5.将EffNet应用到方向盘角度预测<br>
使用神经网络了预测方向盘转动角度是我另外一个项目（[传送门](https://github.com/LeeWise9/Autopilot_Self_Driving_Car_Steering)），本项目考虑使用EffNet优化一下该网络。

在构建预测方向盘转动角度的专用Effnet时，采取了和mnist相似的策略，不同的是，增加了批正则化层（BatchNormalization）并把relu激活函数改为LeakyReLU，事实证明，这些举措是挺有效的。

base_model参数量：865,921；模型大小：10,225KB；经21个epochs后：train_loss: 0.1016，val_loss: 0.0994；误差下降曲线如下：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/train_val_loss_base.jpg" alt="Sample"  width="500">
</p>

eff_model参数量：630,433；模型大小：7,566KB；经21个epochs后：train_loss: 0.1224，val_loss: 0.1074；误差下降曲线如下：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/train_val_loss_eff2.jpg" alt="Sample"  width="500">
</p>

EffNet的训练、测试误差值看起来较大，但模型的表现效果更好，我用模拟器录制了一段视频，可以说车技了得！<br>

<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/drive_eff.gif" alt="Sample"  width="500">
</p>

对比两个模型各项参数，可以发现提升还是很明显的。足以证明EffNet在为模型减重，提升模型性能上还是很有效的。<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E5%AF%B9%E6%AF%94.jpg" alt="Sample"  width="500">
</p>
