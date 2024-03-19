import torch
from torch import nn

# 定义一个MyLeNet5的类，继承自nn.module(神经网络模型基类)
class MyLeNet5(nn.Module):
    def __init__(self):#初始化函数，定义网络的各个层和参数
        super(MyLeNet5,self).__init__()#调用父类nn.Module的初始化函数

        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)  #定义一个卷积层c1,输入通道数为1，输出通道数6，卷积核大小5*5，padding为2

        self.Sigmoid=nn.Sigmoid()# 这一行定义了一个Sigmoid激活函数

        self.s2=nn.AvgPool2d(kernel_size=2,stride=2)# 这一行定义了一个2x2的平均池化层s2，用于下采样。

        self.c3=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)#这一行定义了第二个卷积层 c3，输入通道数为6，输出通道数为16，卷积核大小为5x5。

        self.s4=nn.AvgPool2d(kernel_size=2,stride=2)#这一行定义了第二个2x2的平均池化层 s4。

        self.c5=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)#这一行定义了第三个卷积层 c5，输入通道数为16，输出通道数为120，卷积核大小为5x5。

        self.flatten=nn.Flatten() #这一行定义了一个展平层，用于将多维输入展平为一维。
        self.f6=nn.Linear(120,84) #这一行定义了一个全连接层 f6，输入大小为120，输出大小为84。
        self.output=nn.Linear(84,10) #这一行定义了一个输出层，将全连接层的输出映射到10个类别上。

    def forward(self,x): #前向传播函数，定义了数据在网络中的流动路径
        x=self.Sigmoid(self.c1(x))
        x=self.s2(x)
        x=self.Sigmoid(self.c3(x))
        x=self.s4(x)
        x=self.c5(x)
        x=self.flatten(x)
        x=self.f6(x)
        x=self.output(x)
        return x

    # 数据x通过c1卷积层，并经过Sigmoid激活函数。
    # 经过s2平均池化层进行下采样。
    # 数据再经过c3卷积层，并再次经过Sigmoid激活函数。
    # 再通过s4平均池化层进行下采样。
    # 数据再经过c5卷积层。
    # 最后通过展平层将多维数据展平成一维。
    # 数据经过f6全连接层，再经过输出层output。

if __name__=="__main__":
    x=torch.rand([1,1,28,28])  #这一行创建了一个随机张量 x，形状为 [1,1,28,28]，表示一个1通道、28x28大小的图片。
    model=MyLeNet5()#这一行创建了一个 MyLeNet5 类的实例，即一个LeNet-5模型
    y=model(x) #这一行将输入 x 通过LeNet-5模型前向传播，得到输出 y。


