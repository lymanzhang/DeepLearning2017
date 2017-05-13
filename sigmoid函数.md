## sigmoid函数

Sigmoid函数是一个常见的S型的函数，也称为S型生长曲线。

### Sigmoid函数由下列公式定义

其对x的导数可以用自身表示:
S(x) = 1 / 1 + e^-x

$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$  
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)

其对x的导数为
S'(x) = e^-x / (1 + e ^-x)^2 = S(x)(1-S(x))

- 1.  Han, Jun; Morag, Claudio ．The influence of the sigmoid function parameters on the speed of backpropagation learning：From Natural to Artificial Neural Computation.，1995： 195–201

## sigmoid函数在神经网络中的应用
- 1-对于深度神经网络，中间的隐层的输出必须有一个激活函数。否则多个隐层的作用和没有隐层相同。这个激活函数不一定是sigmoid，常见的有sigmoid、tanh、relu等。
- 2-对于二分类问题，输出层是sigmoid函数。这是因为sigmoid函数可以把实数域光滑的映射到[0,1]空间。函数值恰好可以解释为属于正类的概率（概率的取值范围是0~1）。另外，sigmoid函数单调递增，连续可导，导数形式非常简单，是一个比较合适的函数
- 3-对于多分类问题，输出层就必须是softmax函数了。softmax函数是sigmoid函数的推广
