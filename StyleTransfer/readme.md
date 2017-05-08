## 风格迁移
风格迁移是深度学习模型比较具有代表性的应用实现之一，通过风格迁移，你可以按照著名的画作重新创建风格一样的图片！

先来研究一个叫做[快速风格迁移](https://github.com/lengstrom/fast-style-transfer)的模型，首先通过神经网络会学习指定绘画作品的风格，然后再将这一风格应用到对另一图片的处理上，从而生成与指定绘画作品风格非常一致的处理输出。也就是说，这个风格迁移模型能够根据给定画作的风格进行了训练，然后将这些风格迁移到其他图片上，甚至[视频](https://www.youtube.com/watch?v=xVJwwWQlQ1o)上！

比如本例中，分别采用了来自5个不同风格的绘画作品进行训练，并应用到同一对象上。

### 原参考风格
### la_muse风格 [Pablo Picasso](https://en.wikipedia.org/wiki/Pablo_Picasso)  
![la_muse风格](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/style/la_muse.jpg)
### rain-princess风格 [Leonid Afremov](https://afremov.com/Leonid-Afremov-bio.html)  
![rain_princess风格](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/style/rain_princess.jpg)
### wreck风格 [J.M.W. Turner](https://en.wikipedia.org/wiki/J._M._W._Turner)  
![wreck风格](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/style/the_shipwreck_of_the_minotaur.jpg)
### udnie风格 [Francis Picabia](https://en.wikipedia.org/wiki/Francis_Picabia)  
![udnie风格](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/style/udnie.jpg)
### wave风格 [Hokusai](https://en.wikipedia.org/wiki/Hokusai)  
![wave风格](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/style/wave.jpg)


### 原迁移目标图
![Trump](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02.jpeg)

### la_muse风格
![Trump_la_muse](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_la_muse.jpg)

### rain-princess风格
![Trump_rain-princess](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_rain-princess.jpg)

### udnie风格
![Trump_udnie](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_udnie.jpg)

### wave风格
![Trump_wave](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_wave.jpg)

### wreck风格
![Trump_wreck](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Trump/cp02_output_wreck.jpg)

要自己试一下，你可以在 [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer) GitHub 资源库 中找到相关代码。你可以使用 git 克隆该资源库，或将整个资源库下载为 Zip 归档文件，并解压。

自己训练神经网络需要非常大的资源开销，在此我们先使用他人已经训练出来的检查点文件对目标图片进行风格迁移尝试。

该神经网络按照[此处](https://github.com/lengstrom/fast-style-transfer/tree/master/examples/style)的几种不同风格进行了训练，并保存在[检查点文件](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ)中。检查点文件包含了关于已训练神经网络的所有信息，可以将风格应用到新的图片上。

### 依赖项

- Windows
对于 Windows，你需要安装 TensorFlow 0.12.1、Python 3.5、Pillow 3.4.2、scipy 0.18.1 和 numpy 1.11.2。安装 Miniconda 后，打开你的命令提示符窗口。然后逐行输入以下命令：
```
conda create -n style-transfer python=3.5
activate style-transfer
pip install tensorflow
conda install scipy pillow
```
第一行创建一个新的环境，其中存储了格式迁移代码所需的程序包。第二行 (activate style-transfer) 会进入该环境，你应该会在提示符窗口的开头看到环境名称。接下来的两行负责安装 TensorFlow、Scipy 和 Pillow（一种图像处理库）。

- OS X 和 Linux
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
比如：
```
python evaluate.py --checkpoint ./models/la_muse.ckpt --in-path ./imgs/SBN_04.jpg --out-path ./imgs/SBN_04_output_udnie.jpg
```

### 其他示例

### 原迁移目标图
![TheGate](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/theGate/jiaoda01.jpeg)

### la_muse风格
![TheGate_la_muse](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/theGate/jiaoda01_output_la_muse.jpg)

### rain-princess风格
![TheGate_rain-princess](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/theGate/jiaoda01_output_rain-princess.jpg)

### udnie风格
![TheGate_udnie](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/theGate/jiaoda01_output_udnie.jpg)

### wave风格
![TheGate_wave](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/theGate/jiaoda01_output_wave.jpg)

### wreck风格
![TheGate_wreck](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/theGate/jiaoda01_output_wreck.jpg)


### 原迁移目标图
![Forest](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Forest/SBN_04.jpg)

### la_muse风格
![Forest_la_muse](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Forest/SBN_04_output_la_muse.jpg)

### rain-princess风格
![Forest_rain-princess](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Forest/SBN_04_output_rain-princess.jpg)

### udnie风格
![Forest_udnie](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Forest/SBN_04_output_udnie.jpg)

### wreck风格
![Forest_wreck](https://github.com/lymanzhang/DeepLearning2017/blob/master/StyleTransfer/Forest/SBN_04_output_wreck.jpg)

### checkpoint下载链接
- [Rain Princess checkpoint](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587d1865_rain-princess/rain-princess.ckpt)
- [La Muse checkpoint](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588aa800_la-muse/la-muse.ckpt)
- [Udnie checkpoint](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588aa846_udnie/udnie.ckpt)
- [Scream checkpoint](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588aa883_scream/scream.ckpt)
- [Wave checkpoint](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588aa89d_wave/wave.ckpt)
- [Wreck checkpoint](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588aa8b6_wreck/wreck.ckpt)


## GPU计算的实现

由于GPU计算效率更高，因此部署好GPU计算环境能够很好的加速机器学习速度。  
目前支持GPU计算的显卡均为Nvidia开发，因此无法在Mac pro上部署，本例中GPU计算部署在win10的Dell M7710移动图新工作站上。  
未来计划部署4张基于pascal的TitanX，应该性能更佳。  
本人采用的是DELL 2016年发布的移动工作站m7710，操作系统为win10，至强1535v3的CPU，M5000M的图形卡，Nvidia公布的GPU计算性能在5.3,因此对系统进行了GPU计算环境部署。

- 安装Nvidia的GPU驱动
- 安装cuda程序包，这是一个压缩包，直接解压缩到相关地址即可，如本例：c:\cuda
- 将cuda添加到系统路径
- 通过anaconda创建一个新的python3环境，如styletransfergpu，激活环境，在该环境下安装tensorflow的gpu版本
- 下载VGG19网络模型，该是由牛津视觉几何组（Visual Geometry Group）开发的卷积神经网络结构，它在视觉方面有着不错的表现，项目中也需要用到VGG19网络模型。该文件500多兆，下载速度尚可。
　　[VGG19下载地址](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
- 根据需要运行相关代码，进行模型训练(style.py)或评估(evaluate.py).

实际使用中发现设置为gpu计算后，评估速度大大加快，感觉对于同一张target图，评估时间要快十多倍（感觉，没有用key方法做量化评估）。训练中退出，console提示显示内存不够用（本例实用的是M5000M，显存为8GB，依然不够用），因此训练速度是否有提高尚没有实证，不过应该是显然的。
