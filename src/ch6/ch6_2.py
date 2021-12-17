##
import torch
from torch import nn


##
def corr2d(X, K):  # @save
    """计算二维互相关运算。"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones((6, 8))
X[:, 2:6] = 0
X
# tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.]])

##
K = torch.tensor([[1.0, -1.0]])
# 现在，我们对参数 X （输入）和 K （卷积核）执行互相关运算。 如下所示，输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为0 。
Y = corr2d(X, K)
Y
# tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])

##
# 现在我们将输入的二维图像转置，再进行如上的互相关运算。 其输出如下，之前检测到的垂直边缘消失了。 不出所料，这个卷积核K只可以检测垂直边缘，无法检测水平边缘。
corr2d(X.t(), K)
# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])

##
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size = (1, 2), bias = False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

# 在每次迭代中，我们比较 Y 与卷积层输出的平方误差，然后计算梯度来更新卷积核。为了简单起见，我们在此使用内置的二维卷积层，并忽略偏置。
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1}, loss {l.sum():.3f}')
        
# batch 2, loss 6.337
# batch 4, loss 1.832
# batch 6, loss 0.623
# batch 8, loss 0.234
# batch 10, loss 0.092

##
conv2d.weight.data.reshape((1, 2))
# Out[7]: tensor([[ 0.9599, -1.0217]])
