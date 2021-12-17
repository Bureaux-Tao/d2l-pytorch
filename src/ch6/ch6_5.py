##
import torch
from torch import nn
from d2l import torch as d2l

##
X = torch.arange(16, dtype = d2l.float32).reshape((1, 1, 4, 4))
X
# tensor([[[[ 0.,  1.,  2.,  3.],
#           [ 4.,  5.,  6.,  7.],
#           [ 8.,  9., 10., 11.],
#           [12., 13., 14., 15.]]]])

##
pool2d = nn.MaxPool2d(3)
pool2d(X)
# tensor([[[[10.]]]])

##
pool2d = nn.MaxPool2d(3, padding = 1, stride = 2)
pool2d(X)
# tensor([[[[ 5.,  7.],
#           [13., 15.]]]])

##
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
pool2d(X)
