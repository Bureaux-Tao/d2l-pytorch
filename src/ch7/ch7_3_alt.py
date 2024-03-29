##
import copy
import time

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


batch_size = 128
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
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
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
def train(model, traindataloader, train_rate, criterion, optimizer, device, num_epochs = 25,
          saved_name = 'checkpoint.pt'):
    # model:网络模型
    # trainloader:训练数据集，会切分为训练集和验证集
    # train_rate:训练集batchsize百分比
    # criterion:损失函数
    # optimizer:优化方法
    # num_epochs:训练的轮数
    # 计算训练使用的batch数量
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    # 复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    reduceLR = ReduceLROnPlateau(optimizer, factor = 0.25, patience = 2, min_lr = 1e-8, verbose = True)
    early_stopping = EarlyStopping(patience = 10, verbose = False, save_model = True,
                                   path = path.weight_path + saved_name)
    
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    model.apply(init_weights)
    print('training on', device)
    model.to(device)
    
    since = time.time()
    
    epoch = 0
    for epoch in range(num_epochs):
        # print('Epoch {}/{}:'.format(epoch + 1, num_epochs))
        # print('-' * 10)
        # 每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        
        loop = tqdm(enumerate(traindataloader), total = len(traindataloader))
        for step, (b_x, b_y) in loop:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            if step < train_batch_num:
                model.train()  ##设置模型为训练模式
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm(model.parameters(), max_norm = 1, norm_type = 2)
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(loss = train_loss / train_num,
                                 acc = train_corrects.double().double().item() / train_num)
            else:
                model.eval()  # 3设置模型为评估模式
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        
        early_stopping(val_loss / val_num, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        reduceLR.step(val_loss / val_num)
        
        # 计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('Train Loss:{:.4f}  Train Acc: {:.4f}'.format(train_loss_all[-1], train_acc_all[-1]))
        print('Val Loss:{:.4f}  Val Acc:{:.4f}'.format(val_loss_all[-1], val_acc_all[-1]))
        # 拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    # 使用最好模型的参数
    model.load_state_dict(best_model_wts)
    history = {
        "epoch": range(epoch),
        "loss": train_loss_all,
        "val_loss": val_loss_all,
        "acc": train_acc_all,
        "val_acc": val_acc_all
    }
    return model, history


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


# lr, num_epochs = 0.1, 100
# optimizer_sgd = torch.optim.SGD(net_sgd.parameters(), lr = lr)
# history_sgd = train(net_sgd, train_iter, test_iter, num_epochs, optimizer_sgd, d2l.try_gpu(), 'nin_net_sgd.pt',
#                     val_split = 0.2)
# plot(history_sgd)

##
loss = nn.CrossEntropyLoss()

lr, num_epochs = 0.05, 100
optimizer_sgd = torch.optim.SGD(net.parameters(), lr = lr)

##
history_sgd = train(net, train_iter, 0.8, loss, optimizer_sgd, d2l.try_gpu(), 100, 'nin_net_test.pt')
plot(history_sgd[1])

##
net.load_state_dict(
    torch.load(path.weight_path + "nin_net_test.pt",
               map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

##
test_acc = evaluate_accuracy_gpu(net, test_iter)
print(f'test acc {test_acc:.3f}')
