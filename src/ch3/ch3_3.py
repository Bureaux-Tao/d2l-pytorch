##
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

##
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

##
print(features.shape)
print(labels.shape)


##
def load_array(data_arrays, batch_size, is_train = True):  # @save
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

##
net = nn.Sequential(nn.Linear(2, 1))

##
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

##
loss = nn.MSELoss()

##
trainer = torch.optim.SGD(net.parameters(), lr = 0.01)

##
num_epochs = 5
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# epoch 1, loss 0.593075
# epoch 2, loss 0.011056
# epoch 3, loss 0.000306
# epoch 4, loss 0.000100
# epoch 5, loss 0.000095

##
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
# w的估计误差： tensor([ 0.0003, -0.0003])
# b的估计误差： tensor([2.1935e-05])
