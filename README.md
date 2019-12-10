# Using-EffiNet-to-build-lightweight-neural-network
This project will explore how the efficiency network can improve the calculation efficiency and reduce the size of the model. Taking the steering wheel angle prediction network as an example, it runs on the GPU based on keras.


本项目将探讨[EffNet](https://arxiv.org/abs/1801.06434)是如何提高计算效率并减小模型大小的，该项目以方向盘角度预测网络为例，基于keras，在gpu上运行。


本项目主要包含四个部分：<br>
* 1.空间可分离卷积（Spatial Separable Convolution）；<br>
* 2.深度可分离卷积（Depth Separable Convolution）；<br>
* 3.将EffNet应用到手写数字识别；<br>
* 4.将EffNet应用到方向盘角度预测。<br>


## 1.空间可分离卷积（Spatial Separable Convolution）<br>
Spatial Separable Convolution 将普通卷集核拆分为两个更小的卷积核，例如，最常用的情况是将 3x3 的卷积核拆分为 1x3 和 3x1 的卷积核。

以3x3卷积核为例,输出一个通道,普通卷积需要的参数为32xDxDxM，其中D为输入特征图像的长/宽，M为输入图像的通道数。

若使用空间可分离卷积，先使用1x3的卷积核后使用3x1的卷积核,计算量为(3+3)xDxDxM，参数量为普通卷积的6/9。

推广到一般情况，以nxn(n>2)卷积核为例，计算量从 nxnxDxM 减少到了 (n+n)xDxM。

空间可分离卷积示意图如下：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%A9%BA%E9%97%B4%E5%8F%AF%E5%88%86%E7%A6%BB%E5%8D%B7%E7%A7%AF.png" alt="Sample"  width="500">
</p>


## 2.深度可分离卷积（Depth Separable Convolution）<br>
Depth Separable Convolution 将普通卷积分为两个步骤，下面举例说明。

假设原始图像是二维的，大小是12x12，RGB格式的，有三个通道，相当于一个3维的图片。其输入图片格式是：12x12x3。滤波器窗口大小是5x5x3，得到的输出图像大小是8x8x1（padding模式是valid）。<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%99%AE%E9%80%9A%E5%8D%B7%E7%A7%AF.jpg" alt="Sample"  width="500">
</p>

一个5x5x3滤波器得到的输出图像8x8x1，仅仅提取到的图片里面的一个属性。如果希望获取图片更多的属性，譬如要提取256个属性，则需要：12x12x3 * 5x5x3x256 => 8x8x256个参数。如下图所示：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%99%AE%E9%80%9A%E5%8D%B7%E7%A7%AF2.jpg" alt="Sample"  width="500">
</p>

正常卷积的问题在于，它的卷积核是针对图片的所有通道设计的。那么每要求增加检测图片的一个属性，卷积核就要增加一个。所以普通卷积，卷积参数的总数 = 属性的总数x卷积核的大小。


深度可分离卷积分为两步：<br>
1.用三个卷积对三个通道分别做卷积，这样在一次卷积后，输出3个数。<br>
2.这输出的三个数，再通过一个1x1x3的卷积核，得到一个数。<br>
所以深度可分离卷积其实是通过两次卷积实现的。

第一步，对三个通道分别做卷积，输出三个通道的属性：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%B7%B1%E5%BA%A6%E5%8F%AF%E5%88%86%E7%A6%BB1.jpg" alt="Sample"  width="500">
</p>


第二步，用卷积核1x1x3对三个通道再次做卷积，这个时候的输出就和正常卷积一样，是8x8x1：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%B7%B1%E5%BA%A6%E5%8F%AF%E5%88%86%E7%A6%BB2.jpg" alt="Sample"  width="500">
</p>


如果要提取更多的属性，则需要设计更多的1x1x3卷积核心就可以
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%B7%B1%E5%BA%A6%E5%8F%AF%E5%88%86%E7%A6%BB3.jpg" alt="Sample"  width="500">
</p>

可以看到，如果仅仅是提取一个属性，深度可分离卷积的方法，不如正常卷积。随着要提取的属性越来越多，深度可分离卷积就能够节省更多的参数。





## 3.将EffNet应用到手写数字识别



## 4.将EffNet应用到方向盘角度预测



