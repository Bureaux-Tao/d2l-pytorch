##
import torch
from torch import nn
from torch.nn import functional as F

##
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)


# tensor([[-0.1414,  0.3063, -0.2864, -0.2222,  0.2554,  0.0854, -0.2089,  0.1394,
#          -0.0596,  0.0628],
#         [-0.0630,  0.1263, -0.1068, -0.2003,  0.2523, -0.0326, -0.1521,  0.1189,
#           0.0930,  0.0974]], grad_fn=<AddmmBackward0>)

##
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用`MLP`的父类`Block`的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数`params`（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层
    
    # 定义模型的正向传播，即如何根据输入`X`返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))


net = MLP()
net(X)


# tensor([[-0.1455,  0.3177, -0.1414,  0.0048,  0.1139,  0.0163, -0.0954,  0.1844,
#          -0.1902, -0.0186],
#         [-0.1496,  0.3892, -0.1756, -0.0068,  0.1770,  0.0141, -0.0773, -0.0117,
#          -0.0718, -0.0334]], grad_fn=<AddmmBackward0>)

##
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变。
        self.rand_weight = torch.rand((20, 20), requires_grad = False)
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及`relu`和`dot`函数。
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数。
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
net(X)


##
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
