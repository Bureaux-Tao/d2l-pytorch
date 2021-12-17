##
import random
import torch

##
def synthetic_data(w, b, num_examples):  # @save
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

##
print('features:', features[0], '\nlabel:', labels[0])

##
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
# 一个batch的数据
# tensor([[ 0.3108,  1.7033],
#         [-0.9721,  0.2481],
#         [ 1.4226,  0.3568],
#         [-0.3173, -0.8022],
#         [ 0.6507, -0.5825],
#         [ 0.5864, -0.1881],
#         [-0.9072, -0.4534],
#         [-1.4133,  1.3399],
#         [ 1.1360,  0.4044],
#         [ 0.2289, -0.4320]])
#  tensor([[-0.9649],
#         [ 1.4159],
#         [ 5.8419],
#         [ 6.2825],
#         [ 7.4761],
#         [ 6.0036],
#         [ 3.9190],
#         [-3.1881],
#         [ 5.0989],
#         [ 6.1277]])

##
w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)
w, b


# (tensor([[0.0148],
#          [0.0123]], requires_grad=True),
#  tensor([0.], requires_grad=True))

## 定义模型
def linreg(X, w, b):  # @save
    """线性回归模型。"""
    return torch.matmul(X, w) + b


## 定义均方差损失
# 在实现中，我们需要将真实值y的形状转换为和预测值y_hat的形状相同。
def squared_loss(y_hat, y):  # @save
    """均方损失。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


## 定义优化器
# 在每一步中，使用从数据集中随机抽取的一个小批量，然后根据参数计算损失的梯度。
# 接下来，朝着减少损失的方向更新我们的参数。 下面的函数实现小批量随机梯度下降更新。
# 该函数接受模型参数集合、学习速率和批量大小作为输入。每一步更新的大小由学习速率lr决定。
# 因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（batch_size）来归一化步长，这样步长大小就不会取决于我们对批量大小的选择。
def sgd(params, lr, batch_size):  # @save
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


## 训练
# 在每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测。
# 计算完损失后，我们开始反向传播，存储每个参数的梯度。最后，我们调用优化算法sgd来更新模型参数。
# 在每个迭代周期（epoch）中，我们使用data_iter函数遍历整个数据集，并将训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。
# 这里的迭代周期个数num_epochs和学习率lr都是超参数，分别设为3和0.03。设置超参数很棘手，需要通过反复试验进行调整。
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # `X`和`y`的小批量损失
        # 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# epoch 1, loss 0.027381
# epoch 2, loss 0.000090
# epoch 3, loss 0.000048

##
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
# w的估计误差: tensor([-0.0006, -0.0001], grad_fn=<SubBackward0>)
# b的估计误差: tensor([0.0008], grad_fn=<RsubBackward1>)
