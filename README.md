# Using-EffNet-to-build-lightweight-neural-network
This project will explore how the efficiency network can improve the calculation efficiency and reduce the size of the model. Taking the steering wheel angle prediction network as an example, it runs on the GPU based on keras.


本项目将探讨[EffNet](https://arxiv.org/abs/1801.06434)是如何提高计算效率并减小模型大小的，该项目以方向盘角度预测网络为例，基于keras，在gpu上运行。


本项目主要包含四个部分：<br>
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

一个5x5x3滤波器得到的输出图像8x8x1，设通道数为256，不考虑偏置项，需要：5x5x5x256 = 19200 个参数，计算量为：5x5x3x256x8x8 = 1228800 。如下图所示：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%99%AE%E9%80%9A%E5%8D%B7%E7%A7%AF2.jpg" alt="Sample"  width="500">
</p>

正常卷积的问题在于，卷积核将对图片的所有通道做卷积计算的。那么输出每增加一个通道，卷积核就要增加一个。


深度可分离卷积分为两步：<br>
>1.用三个卷积对三个通道分别做卷积，这样在一次卷积后，输出3个数。<br>
>2.这输出的三个数，再通过一个1x1x3的卷积核，得到一个数。<br>
所以深度可分离卷积其实是通过两次卷积实现的。

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
将EffNet应用到手写数字识别。


## 5.将EffNet应用到方向盘角度预测<br>
将EffNet应用到方向盘角度预测。


