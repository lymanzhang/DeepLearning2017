## 风格迁移
风格迁移是深度学习模型比较具有代表性的应用实现之一，通过风格迁移，你可以按照著名的画作重新创建风格一样的图片！

先来研究一个叫做快速风格迁移的模型，首先通过神经网络会学习指定绘画作品的风格，然后再将这一风格应用到对另一图片的处理上，从而生成与指定绘画作品风格非常一致的处理输出。也就是说，这个风格迁移模型能够根据给定画作的风格进行了训练，然后将这些风格迁移到其他图片上，甚至视频上！

比如本例中，分别采用了来自5个不同风格的绘画作品进行训练，并应用到同一对象上。

### 原参考风格

### 原迁移目标图
![Trump](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02.jpeg)

### la_muse风格
![Trump_la_muse](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_la_muse.jpg)

### rain-princess风格
![Trump_rain-princess](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_rain-princess.jpg)

### udnie风格
![Trump_udnie](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_udnie.jpg)

### wave风格
![wave](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_wave.jpg)

### wreck风格
![Trump_wreck](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_wreck.jpg)

要自己试一下，你可以在 [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer) GitHub 资源库 中找到相关代码。你可以使用 git 克隆该资源库，或将整个资源库下载为 Zip 归档文件，并解压。

该神经网络按照[此处](https://github.com/lengstrom/fast-style-transfer/tree/master/examples/style)的几种不同风格进行了训练，并保存在[检查点文件](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ)中。检查点文件包含了关于已训练神经网络的所有信息，可以将风格应用到新的图片上。

依赖项
安装运行该代码所需的所有程序包的最简单方式是使用 Miniconda，它是更简版本的 Anaconda。 Miniconda 自带了 Conda，后者是专门用于数据科学的程序包和环境管理器。请安装适合你的操作系统的 Python 3 版本的 Miniconda。

如果你之前没有使用过 Conda，请快速阅读我们的教程。

Windows
对于 Windows，你需要安装 TensorFlow 0.12.1、Python 3.5、Pillow 3.4.2、scipy 0.18.1 和 numpy 1.11.2。安装 Miniconda 后，打开你的命令提示符窗口。然后逐行输入以下命令：

conda create -n style-transfer python=3.5
activate style-transfer
pip install tensorflow
conda install scipy pillow
第一行创建一个新的环境，其中存储了格式迁移代码所需的程序包。第二行 (activate style-transfer) 会进入该环境，你应该会在提示符窗口的开头看到环境名称。接下来的两行负责安装 TensorFlow、Scipy 和 Pillow（一种图像处理库）。

OS X 和 Linux
对于 OS X 和 Linux，你需要安装 TensorFlow 0.11.0、Python 2.7.9、Pillow 3.4.2、scipy 0.18.1 和 numpy 1.11.2.

在终端里，逐行输入以下命令：
```
conda create -n style-transfer python=2.7.9
source activate style-transfer
pip install tensorflow
conda install scipy pillow
```

第一行创建一个新的环境，其中存储了格式迁移代码所需的程序包。第二行 (source activate style-transfer) 进入该环境，你应该会在提示符窗口的开头看到环境名称。接下来的两行负责安装 TensorFlow、Scipy 和 Pillow（一种图像处理库）。

### 迁移风格
- 从[fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)资源库中下载 Zip 归档文件并解压。你可以通过点击右侧的亮绿色按钮进行下载。
- 从[此处](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587d1865_rain-princess/rain-princess.ckpt)下载 Rain Princess 检查点，将其放在 fast-style-transfer 文件夹中。检查点文件是已经调谐参数的模型。使用此检查点文件后我们不需要再训练该模型，可以直接使用。
- 将你要调整格式的图片放到 fast-style-transfer 文件夹。
- 进入你之前创建的 Conda 环境（如果不在里面的话）。
- 在终端里，转到 fast-style-transfer 文件夹并输入:
```
python evaluate.py --checkpoint ./rain-princess.ckpt --in-path <path_to_input_file> --out-path ./output_image.jpg
```
