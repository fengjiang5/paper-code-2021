# paper-code-2021
<font face="Times New Roman" size=4>*This file is the code of undergraduate graduation design*</font>:bowtie:


# Description of The File：  
## 1. 环境说明：  
&#8195;&#8195;4.2节所进行实验所用硬件设备为Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz   2.50 GHz，NIVIDA GeForce GTX 1050Ti；编译软件为：PyCharm 2019.3.5，Python 3.7.9，核心包及其版本为：torchvision 0.3.0，Pytorch 1.1.0。  
&#8195;&#8195;4.3节所进行实验所用硬件设备为Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz   2.50 GHz，实验在Google Colaboratory上使用GPU进行，型号为Tesla K80，另外其他Python附加包均为直接预装在Google Drive上的最新版本。  
## 2. 文件说明：  
&#8195;&#8195;文件4.2当中包含的[main.py](https://github.com/fengjiang5/paper-code-2021/blob/main/jf%20paper%20code/4.2/main.py)用于进行卷积神经网络与全连接网络的对比实验。文件可直接运行。  
&#8195;&#8195;文件4.3用于进行经典卷积神经网络的对比实验，其中的[AlexNet_VGG.ipynb](https://github.com/fengjiang5/paper-code-2021/blob/main/jf%20paper%20code/4.3/AlexNet_VGG%20.ipynb)包含AlexNet与VGG，可以使用Google Colab在线打开也可使用本地jupyter notebook打开；[Googlenet.ipynb](https://github.com/fengjiang5/paper-code-2021/blob/main/jf%20paper%20code/4.3/Googlenet.ipynb)包含GoogLeNet的Inception1，打开方式同上；[Resnet18.ipynb](https://github.com/fengjiang5/paper-code-2021/blob/main/jf%20paper%20code/4.3/Res18.ipynb)包含ResNet18模型，打开方式同上。  
&#8195;&#8195;文件4.4用于进行残差网络的深度增加实验，[Resnet.ipynb](https://github.com/fengjiang5/paper-code-2021/blob/main/jf%20paper%20code/4.4/res/Resnet.ipynb)包含所需五种模型的定义，并且可以直接训练测试，Loss/Accuracy会直接保存在生成的txt文档中。*注：此代码运行时间较长*&#8195;from:clock8:to:clock4:  
&#8195;&#8195;文件4.5包含用于可视化比较优化方法的[optimizar.ipynb](https://github.com/fengjiang5/paper-code-2021/blob/main/jf%20paper%20code/4.5/optimizar.ipynb)，文件插入了包ipywidgets来交互地调整参数。方法及效果如下。
```Python
from ipywidgets import *
```
```
@interact(lr=(0, 1, 0.001),epoch=(0,100,1),init_x1=(-5,5,0.1),init_x2=(-5,5,0.1),continuous_update=False)
```
![optim](https://github.com/fengjiang5/paper-code-2021/blob/main/jf%20paper%20code/4.5/optim.png)
