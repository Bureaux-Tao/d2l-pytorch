##
import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l

##
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train = False)


##
def init_params():
    w = torch.normal(0, 1, size = (num_inputs, 1), requires_grad = True)
    b = torch.zeros(1, requires_grad = True)
    return [w, b]


##
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


##
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel = 'epochs', ylabel = 'loss', yscale = 'log',
                            xlim = [5, num_epochs], legend = ['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                # 增加了L2范数惩罚项，广播机制使l2_penalty(w)成为一个长度为`batch_size`的向量。
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())


##
train(lambd = 0)
# w的L2范数是： 13.97132396697998

##
train(lambd = 3)


# w的L2范数是： 0.3416399657726288

##
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减。
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}], lr = lr)
    animator = d2l.Animator(xlabel = 'epochs', ylabel = 'loss', yscale = 'log',
                            xlim = [5, num_epochs], legend = ['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())


plt.show()

##
train_concise(0)

# l2正则
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
# or
# optimizer = torch.optim.Adam([{'params': model.base.parameters(), 'lr':1e-3},
#                               {'params':model.classifier.parameters(),'lr':1e-4}],
#                               lr=1e-2, momentum=0.9, weight_decay=1e-5)
# 一般正则化，只是对模型的权重W参数进行惩罚，而偏置参数b是不进行惩罚的，而torch.optim的优化器weight_decay参数指定的权值衰减是对网络中的所有参数，包括权值w和偏置b同时进行惩罚。很多时候如果对b 进行L2正则化将会导致严重的欠拟合，因此这个时候一般只需要对权值w进行正则即可。