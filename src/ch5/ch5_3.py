##
import torch
import torch.nn.functional as F
from torch import nn


##
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
# Out[30]: tensor([-2., -1.,  0.,  1.,  2.])

##
net = nn.Sequential(nn.Linear(8, 16), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y)
Y.mean()


# tensor([[ 0.1876,  0.4416,  0.0642, -0.1712,  0.1136, -0.4535, -0.1479,  0.0865,
#          -0.3968,  0.1882,  0.3080,  0.2498,  0.0884,  0.5405, -0.0973, -0.0335],
#         [ 0.2942,  0.6069,  0.1236, -0.4507, -0.1108, -0.4064, -0.0822,  0.1770,
#          -0.4898,  0.5280,  0.1484, -0.2725, -0.2424,  0.2049, -0.1607, -0.3245],
#         [ 0.0569,  0.6588, -0.3031, -0.5076, -0.6124, -0.2285,  0.0073, -0.1194,
#          -0.5053,  0.5733,  0.2300, -0.0637,  0.1042,  0.8195, -0.2698, -0.3674],
#         [-0.0372,  0.6369, -0.1840, -0.2850, -0.3304, -0.3724, -0.1502, -0.0084,
#          -0.5527,  0.3601,  0.2208,  0.1658,  0.1769,  0.9639, -0.4185, -0.1696]],
#        grad_fn=<SubBackward0>)
# Out[32]: tensor(-4.6566e-09, grad_fn=<MeanBackward0>)

##
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))
    
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


##
linear = MyLinear(5, 3)
print(linear.weight)
# Parameter containing:
# tensor([[-1.6903, -2.2195, -1.3282],
#         [-0.5830, -0.0561,  1.8252],
#         [-1.3627, -0.6810, -1.2960],
#         [ 0.3635, -0.9873, -0.9140],
#         [-0.6028,  0.6294, -1.8759]], requires_grad=True)

##
linear(torch.rand(2, 5))
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

##
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
# tensor([[0.],
#         [0.]])
