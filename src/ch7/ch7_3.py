##
import copy

import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append("/home/bureaux/Projects/TorchProject/")
sys.path.append("/home/bureaux/Projects/TorchProject/d2l/")
import path
from d2l.earlystopping import EarlyStopping

# os.environ["OMP_NUM_THREADS"] = "1"
from d2l import torch as d2l


##
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 1), nn.ReLU())


net = nn.Sequential(
    nin_block(1, 96, kernel_size = 11, strides = 4, padding = 0),
    nn.MaxPool2d(3, stride = 2),
    nin_block(96, 256, kernel_size = 5, strides = 1, padding = 2),
    nn.MaxPool2d(3, stride = 2),
    nin_block(256, 384, kernel_size = 3, strides = 1, padding = 1),
    nn.MaxPool2d(3, stride = 2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size = 3, strides = 1, padding = 1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    nn.Flatten(),
    # nn.Linear(128, 32),
    # nn.ReLU(),
    # nn.Dropout(0.5),
    # nn.Linear(32, 10),
)

net_sgd = copy.deepcopy(net)
net_adam = copy.deepcopy(net)

X = torch.randn(size = (1, 1, 224, 224))
print(net)
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__.ljust(15, " "), 'output shape:\t', X.shape)


##
def load_data_fashion_mnist(batch_size, resize = None, root = 'data'):
    trans = [transforms.ToTensor(), transforms.Normalize(mean = [0.5], std = [0.5])]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = root,
                                                    train = True,
                                                    transform = trans,
                                                    download = True)
    mnist_test = torchvision.datasets.FashionMNIST(root = root,
                                                   train = False,
                                                   transform = trans,
                                                   download = True)
    return (DataLoader(mnist_train, batch_size, shuffle = True,
                       num_workers = 2),
            DataLoader(mnist_test, batch_size, shuffle = False,
                       num_workers = 2))


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize = 224,
                                                root = path.data_path)


##
class Accumulator:  # @save
    """在`n`个变量上累加。"""
    
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device = None):  # @save
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def evaluate_accuracy_cpu(net, data_iter):  # @save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


##
# @save
def train(net, train_iter, test_iter, num_epochs, optimizer, device, saved_name, val_split = 0.2):
    """用GPU训练模型(在第六章定义)。"""
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    
    batch_num = len(train_iter)
    train_batch_num = round(batch_num * float(1 - val_split))
    
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    reduceLR = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3, min_lr = 1e-8, verbose = True)
    early_stopping = EarlyStopping(patience = 10, verbose = False, save_model = True,
                                   path = path.weight_path + saved_name)
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = d2l.Accumulator(3)
        net.train()
        
        loop = tqdm(enumerate(train_iter), total = len(train_iter))
        for i, (X, y) in loop:
            if i < train_batch_num:
                net.train()
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                nn.utils.clip_grad_norm(net.parameters(), max_norm = 3, norm_type = 2)
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                #     animator.add(epoch + (i + 1) / num_batches,
                #                  (train_l, train_acc, None))
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(loss = train_l, acc = train_acc)
            else:
                net.eval()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                val_l = metric[0] / metric[2]
                val_acc = metric[1] / metric[2]
        
        timer.stop()
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'val loss {val_l:.3f}, val acc {val_acc:.3f}, ')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')
        train_loss_all.append(train_l)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_l)
        val_acc_all.append(val_acc)
        early_stopping(val_l, net)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        reduceLR.step(val_l)
    print('\n-------------Test Set Evaluation----------')
    test_acc = evaluate_accuracy_gpu(net, test_iter)
    # animator.add(epoch + 1, (None, None, test_acc))
    print(f'test acc {test_acc:.3f}')
    return {
        "epoch": range(num_epochs),
        "loss": train_loss_all,
        "acc": train_acc_all,
        "val_acc": val_acc_all,
        "val_loss": val_loss_all
    }


##
def plot(history):
    train_process = pd.DataFrame(history)
    
    # 设置value的显示长度为200，默认为50
    pd.set_option('max_colwidth', 200)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    # 显示所有行，把列显示设置成最大
    pd.set_option('display.max_rows', None)
    
    print(train_process)
    
    # 可视化模型训练过程
    plt.rcParams['font.sans-serif'] = ['SF Mono']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    
    plt.figure(figsize = (12, 4))
    # 损失函数
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.loss, "ro-", label = "Train loss")
    plt.plot(train_process.epoch, train_process.val_loss, "ro-", label = "Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    # 精度
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.acc, "ro-", label = "Train acc")
    plt.plot(train_process.epoch, train_process.val_acc, "bs-", label = "Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


lr, num_epochs = 0.1, 100
optimizer_sgd = torch.optim.SGD(net_sgd.parameters(), lr = lr)
history_sgd = train(net_sgd, train_iter, test_iter, num_epochs, optimizer_sgd, d2l.try_gpu(), 'nin_net_sgd.pt',
                    val_split = 0.2)
plot(history_sgd)

lr, num_epochs = 0.001, 100
optimizer_adam = torch.optim.Adam(net_adam.parameters(), lr = lr)
history_adam = train(net_adam, train_iter, test_iter, num_epochs, optimizer_adam, d2l.try_gpu(), 'nin_net_adam.pt',
                     val_split = 0.2)
plot(history_adam)

##
test_acc = evaluate_accuracy_gpu(net_adam, test_iter)
print(f'test acc {test_acc:.3f}')
