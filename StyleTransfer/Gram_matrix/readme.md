### [关于style transfer中Gram matrix的应用](https://www.zhihu.com/question/49805962/answer/157003898)

style transfer 思路很简单：在图像内容附近通过白噪声初始化一个输出的结果，然后通过网络对这个结果进行风格和内容两方面的约束进行修正。而在风格的表示中采用的是Gram Matrix。我是这样理解为什么用Gram 矩阵的：度量各个维度自己的特性以及各个维度之间的关系。  style transfer 当中，什么是风格，存在自己特性的才叫做风格。因此如何去度量这个自己的特性勒，自己的特点越突出，别人的越不突出最好。因此论文当中这样去做：

![fumula1](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Gram_matrix/01.png)

这样我们知道：当同一个维度上面的值相乘的时候原来越小酒变得更小，原来越大就变得越大；二不同维度上的关系也在相乘的表达当中表示出来。

![fumula2](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Gram_matrix/02.png)

因此，最终能够在保证内容的情况下，进行风格的传输。

补充一下。内积之后得到的多尺度矩阵中，对角线元素提供了不同特征图（a1，a2 ... ，an）各自的信息，其余元素提供了不同特征图之间的相关信息。于是，在一个Gram矩阵中，既能体现出有哪些特征，又能体现出不同特征间的紧密程度。论文中作者把这个定义为风格。

content 是关于图像中物体在整幅图像的分布，因此让high-level content（即high-level layer输出）接近，就可以得到和原图类似的content。style是图像的纹理信息（比较抽象），纹理信息在high-level中的表示就是features map值之间的相关性。features map是矩阵（可以看做向量），而Gram matrix是用来衡量向量相关程度的。$G_{ij}^l$就是第$l$layer输出的$i$ feature map和 $j$feature map的内积。

