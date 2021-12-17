##
import torch

##
x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x ** y
# (tensor([5.]), tensor([6.]), tensor([1.5000]), tensor([9.]))

##
x = torch.arange(4)
x[3]
# tensor(3)

##
len(x)

##
A = torch.arange(20).reshape(5, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19]])

A.T  # 转置
# tensor([[ 0,  4,  8, 12, 16],
#         [ 1,  5,  9, 13, 17],
#         [ 2,  6, 10, 14, 18],
#         [ 3,  7, 11, 15, 19]])

##
A = torch.arange(20, dtype = torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A * B
# tensor([[  0.,   1.,   4.,   9.],
#         [ 16.,  25.,  36.,  49.],
#         [ 64.,  81., 100., 121.],
#         [144., 169., 196., 225.],
#         [256., 289., 324., 361.]])

## 数乘
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
# (tensor([[[ 2,  3,  4,  5],
#           [ 6,  7,  8,  9],
#           [10, 11, 12, 13]],
#
#          [[14, 15, 16, 17],
#           [18, 19, 20, 21],
#           [22, 23, 24, 25]]]),
#  torch.Size([2, 3, 4]))

##
x = torch.arange(4, dtype = torch.float32)
x, x.sum()
# (tensor([0., 1., 2., 3.]), tensor(6.))
A_sum_axis0 = A.sum(axis = 0)
A, A_sum_axis0, A_sum_axis0.shape
# (tensor([[ 0.,  1.,  2.,  3.],
#          [ 4.,  5.,  6.,  7.],
#          [ 8.,  9., 10., 11.],
#          [12., 13., 14., 15.],
#          [16., 17., 18., 19.]]),
#  tensor([40., 45., 50., 55.]),
#  torch.Size([4]))

##
A.sum(axis = [0, 1])  # Same as `A.sum()`
# tensor(190.)

##
A.mean(axis = 0), A.sum(axis = 0) / A.shape[0]
# (tensor([ 8.,  9., 10., 11.]), tensor([ 8.,  9., 10., 11.]))

##
sum_A = A.sum(axis = 1, keepdims = True)  # 保持维度
sum_A
# tensor([[ 6.],
#         [22.],
#         [38.],
#         [54.],
#         [70.]])

##
A.cumsum(axis = 0)  # 滚动相加
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  6.,  8., 10.],
#         [12., 15., 18., 21.],
#         [24., 28., 32., 36.],
#         [40., 45., 50., 55.]])

##
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)  # 向量点积
# (tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))
##
A, x, torch.mv(A, x)  # 第一个参数是矩阵，第二个参数只能是一维向量，等价于等价于Ax^T
# (tensor([[ 0.,  1.,  2.,  3.],
#          [ 4.,  5.,  6.,  7.],
#          [ 8.,  9., 10., 11.],
#          [12., 13., 14., 15.],
#          [16., 17., 18., 19.]]),
#  tensor([0., 1., 2., 3.]),
#  tensor([ 14.,  38.,  62.,  86., 110.]))

##
B = torch.ones(4, 3)
# 矩阵乘法
A, B, torch.mm(A, B)
# (tensor([[ 0.,  1.,  2.,  3.],
#          [ 4.,  5.,  6.,  7.],
#          [ 8.,  9., 10., 11.],
#          [12., 13., 14., 15.],
#          [16., 17., 18., 19.]]),
#  tensor([[1., 1., 1.],
#          [1., 1., 1.],
#          [1., 1., 1.],
#          [1., 1., 1.]]),
#  tensor([[ 6.,  6.,  6.],
#          [22., 22., 22.],
#          [38., 38., 38.],
#          [54., 54., 54.],
#          [70., 70., 70.]]))

##
u = torch.tensor([3.0, -4.0])
torch.norm(u) # l2范数
# tensor(5.)

##
torch.abs(u).sum() # l1范数
# tensor(7.)

##
torch.norm(torch.ones((4, 9))) # 弗罗贝尼乌斯范数
# tensor(6.)

