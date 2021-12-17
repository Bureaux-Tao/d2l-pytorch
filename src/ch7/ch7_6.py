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
import prettytable as pt

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
class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv = False, strides = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size = 3, padding = 1, stride = strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size = 3, padding = 1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, X):
        route_1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2
        )
        route_2 = nn.Sequential(
            self.conv3
        )
        Y = route_1(X)
        if self.conv3:
            X = route_2(X)
        Y += X
        
        # Y = F.relu(self.bn1(self.conv1(X)))
        # Y = self.bn2(self.conv2(Y))
        # if self.conv3:
        #     X = self.conv3(X)
        # Y += X
        return self.relu(Y)


##
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv = True, strides = 2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
    nn.Sequential(*resnet_block(64, 64, 2, first_block = True)),
    nn.Sequential(*resnet_block(64, 128, 2)),
    nn.Sequential(*resnet_block(128, 256, 2)),
    nn.Sequential(*resnet_block(256, 512, 2)),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
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
def train(model, traindataloader, train_rate, criterion, optim, device, lr = 0.01, num_epochs = 25,
          kernel_regularizer = 0, saved_path = 'checkpoint.pt'):
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
    
    if os.path.isfile(path.weight_path + saved_path):
        print('load weight from ' + path.weight_path + saved_path)
        model.load_state_dict(torch.load(path.weight_path + saved_path, map_location = torch.device(device)))
    else:
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        
        model.apply(init_weights)
    
    if kernel_regularizer > 0:
        optimizer = optim(model.parameters(), lr = lr, weight_decay = kernel_regularizer)
    else:
        optimizer = optim(model.parameters(), lr = lr)
    
    reduceLR = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 2, min_lr = 1e-8, verbose = True)
    early_stopping = EarlyStopping(patience = 5, verbose = False, save_model = True,
                                   path = path.weight_path + saved_path)
    
    print('training on', device)
    model.to(device)
    
    # since = time.time()
    
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
                loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
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
        
        # 计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        # print('Train Loss:{:.4f}  Train Acc: {:.4f}'.format(train_loss_all[-1], train_acc_all[-1]))
        # print('Val Loss:{:.4f}  Val Acc:{:.4f}'.format(val_loss_all[-1], val_acc_all[-1]))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # print(f'Test acc:{test_acc:.3f}')
        # 拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        # time_use = time.time() - since
        # print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
        tb = pt.PrettyTable()
        tb.field_names = ["epoch", "loss", "val loss", "acc", "val acc", "test acc"]
        tb.add_row(
            [str(epoch + 1), str(round(train_loss_all[-1], 5)), str(round(val_loss_all[-1], 5)),
             str(round(train_acc_all[-1], 5)), str(round(val_acc_all[-1], 5)), str(round(test_acc, 5))])
        print(tb)
        early_stopping(val_loss / val_num, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        reduceLR.step(val_loss / val_num)
    
    # 使用最好模型的参数
    model.load_state_dict(best_model_wts)
    history = {
        "epoch": range(epoch) if early_stopping.early_stop else range(epoch + 1),
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

lr, num_epochs = 0.01, 7
optimizer_sgd = torch.optim.SGD

##
history = train(net, train_iter, 0.8, loss, optimizer_sgd, d2l.try_gpu(), lr = lr, kernel_regularizer = 0,
                num_epochs = num_epochs, saved_path = 'resnet.pt')

##
plot(history[1])

##
net.load_state_dict(
    torch.load(path.weight_path + "resnet.pt",
               map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

##
# test_acc = evaluate_accuracy_gpu(net, test_iter)
# print(f'test acc {test_acc:.3f}')

if torch.cuda.is_available():
    net.to("cuda:0" if torch.cuda.is_available() else "cpu")  # 移动模型到cuda
net.eval()
loop = tqdm(enumerate(test_iter), total = len(test_iter))
test_loss = 0.0
test_corrects = 0
test_num = 0
test_loss_all = []
test_acc_all = []
for step, (b_x, b_y) in loop:
    b_x = b_x.cuda() if torch.cuda.is_available() else b_x
    b_y = b_y.cuda() if torch.cuda.is_available() else b_y
    net.eval()  # 3设置模型为评估模式
    output = net(b_x)
    pre_lab = torch.argmax(output, 1)
    l = loss(output, b_y)
    test_loss += l.item() * b_x.size(0)
    test_corrects += torch.sum(pre_lab == b_y.data)
    test_num += b_x.size(0)

test_loss_all.append(test_loss / test_num)
test_acc_all.append(test_corrects.double().item() / test_num)
print('Test Loss:{:.4f}  Test Acc:{:.4f}'.format(test_loss_all[-1], test_acc_all[-1]))
