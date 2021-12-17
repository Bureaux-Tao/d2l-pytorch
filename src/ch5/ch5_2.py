##
import torch
from torch import nn

##
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size = (2, 4))
net(X)
# tensor([[-0.1571],
#         [-0.1828]], grad_fn=<AddmmBackward0>)

##
print(net[0].state_dict())
print(net[1].state_dict())
print(net[2].state_dict())
# OrderedDict([('weight', tensor([[-0.3689,  0.3510,  0.1774, -0.1471],
#         [-0.0860,  0.3281, -0.3878, -0.0147],
#         [-0.3662, -0.1721,  0.1283, -0.2484],
#         [ 0.4774, -0.2119, -0.1179, -0.1721],
#         [ 0.0440, -0.1990,  0.4537,  0.0457],
#         [-0.1926, -0.2008, -0.2346, -0.3468],
#         [-0.4944,  0.4094,  0.1426, -0.3532],
#         [-0.4683, -0.1331,  0.2875, -0.1290]])), ('bias', tensor([-0.4863, -0.2630, -0.3669,  0.0049, -0.0751,  0.0473,  0.0344, -0.1836]))])
# OrderedDict()
# OrderedDict([('weight', tensor([[ 0.0782, -0.0858, -0.3087,  0.2162, -0.3220,  0.0059, -0.1794, -0.0975]])), ('bias', tensor([-0.0923]))])

##
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].bias.data.item())
# <class 'torch.nn.parameter.Parameter'>
# Parameter containing:
# tensor([-0.0923], requires_grad=True)
# tensor([-0.0923])
# -0.09227508306503296

##
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)


# ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
# ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
# tensor([0.1629])

##
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)

##
print(rgnet)
# Sequential(
#   (0): Sequential(
#     (block 0): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 1): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 2): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 3): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#   )
#   (1): Linear(in_features=4, out_features=1, bias=True)
# )

##
print(rgnet[0][1][0].bias.data)


# tensor([-0.2651,  0.2883,  0.4594,  0.3159, -0.3892, -0.2947, -0.1139, -0.3789])

##
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = 0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])
print(net[2].weight.data[0], net[0].bias.data[0])


# tensor([ 0.0059, -0.0098, -0.0011,  0.0096]) tensor(0.)

##
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])


# tensor([1., 1., 1., 1.]) tensor(0.)

##
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
# tensor([-0.6283, -0.5988, -0.5811,  0.0374])
print(net[2].weight.data)


# tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])

##
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(net[0].weight[:2])
# tensor([[ 8.0495, -0.0000, -0.0000, -8.9102],
#         [-8.5649,  0.0000, -8.3750, -0.0000]], grad_fn=<SliceBackward0>)

##
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])
# tensor([42.0000,  1.0000,  1.0000, -7.9102])

##
# 我们需要给共享层一个名称，以便可以引用它的参数。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
print(net(X))
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值。
print(net[2].weight.data[0] == net[4].weight.data[0])
# tensor([[0.0137],
#         [0.0330]], grad_fn=<AddmmBackward0>)
# tensor([True, True, True, True, True, True, True, True])
# tensor([True, True, True, True, True, True, True, True])
