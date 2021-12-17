##
import sys
import os
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torchvision import transforms
import pandas as pd
import time
import copy
import seaborn as sns
from tqdm import tqdm

import path
from d2l.earlystopping import EarlyStopping

os.environ["OMP_NUM_THREADS"] = "1"

##使用FasnionMNIST数据，准备训练数据集
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                transforms.Normalize(mean = [0.5], std = [0.5])])
# 彩图 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
# 灰度图 transforms.Normalize(mean=[0.5],std=[0.5])
BATCH_SIZE = 32

train_data = torchvision.datasets.FashionMNIST(
    root = path.data_path,
    train = True,
    transform = transform,
    download = True,
)
# 定义一个数据加载器
train_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 2,
)
# 计算train_loader有多少个batch
print("train_loader的batch数量为：", len(train_loader))
# train_loader的batch数量为： 1875

##
# 对测试集进行处理
test_data = torchvision.datasets.FashionMNIST(
    root = path.data_path,
    train = False,
    download = True,
    transform = transform
)
test_loader = torch.utils.data.DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 2,
)
##
# 为数据添加一个通道维度,并且取值范围缩放到0-1之间
for i, (images, labels) in enumerate(train_loader):
    print(images.shape)
    break

for i, (images, labels) in enumerate(test_loader):
    print(images.shape)
    break


##
# 卷积神经网络搭建
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


class AlexNet(nn.Module):
    def __init__(self, channel = 3, size = (224, 224)):
        super(AlexNet, self).__init__()
        self.channel = channel
        self.size = size
        # 定义一个卷积层
        self.model = nn.Sequential(
            # 这里，我们使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(1, 96, kernel_size = 11, stride = 4, padding = 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size = 5, padding = 2), nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size = 3, padding = 1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size = 3, padding = 1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Flatten(),
            # nn.Dropout(p = 0.5),
            # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过度拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p = 0.5),
            # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10)
        )
    
    # 定义网络的前向传播路径
    def forward(self, x):
        output = self.model(x)
        return output
    
    def summary(self):
        X = torch.randn(1, self.channel, self.size[0], self.size[1])
        print("Layer name".ljust(15, " "), "Output shape")
        for layer in self.model:
            X = layer(X)
            print(layer.__class__.__name__.ljust(15, " "), X.shape)


# 输出我们的网络结构
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alex_net = AlexNet(channel = 1, size = (224, 224))
print(alex_net)
alex_net.summary()
if torch.cuda.is_available():
    alex_net.to(device)  # 移动模型到cuda


##
# 定义网络的训练过程函数
def train_model(model, traindataloader, train_rate, criterion, optimizer, num_epochs = 25):
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
    reduceLR = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3, min_lr = 1e-8, verbose = True)
    early_stopping = EarlyStopping(patience = 10, verbose = False, save_model = True,
                                   path = '/home/bureaux/Projects/TorchProject/weights/alex_net.pt')
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
            b_x = b_x.cuda() if torch.cuda.is_available() else b_x
            b_y = b_y.cuda() if torch.cuda.is_available() else b_y
            if step < train_batch_num:
                model.train()  ##设置模型为训练模式
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
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
        "train_loss": train_loss_all,
        "val_loss": val_loss_all,
        "train_acc": train_acc_all,
        "val_acc": val_acc_all
    }
    return model, history


##
optimizer = torch.optim.Adam(alex_net.parameters(), lr = 0.001, weight_decay = 0.005)
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()  ##损失函数

##
mynet, history = train_model(
    alex_net, train_loader, 0.8, criterion, optimizer, num_epochs = 200
)

##
train_process = pd.DataFrame(history)
# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)
print(train_process)

##
# 可视化模型训练过程
plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # 图片像素
plt.rcParams['figure.dpi'] = 360  # 分辨率

plt.figure(figsize = (12, 4))
# 损失函数
plt.subplot(1, 2, 1)
plt.plot(train_process.epoch, train_process.train_loss, "ro-", label = "Train loss")
plt.plot(train_process.epoch, train_process.val_loss, "bs-", label = "Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
# 精度
plt.subplot(1, 2, 2)
plt.plot(train_process.epoch, train_process.train_acc, "ro-", label = "Train acc")
plt.plot(train_process.epoch, train_process.val_acc, "bs-", label = "Val acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()

##
# 对测试集进行预测，并可视化预测结果
mynet = AlexNet()
mynet.load_state_dict(
    torch.load("weights/alex_net.pt", map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
if torch.cuda.is_available():
    mynet.to(device)  # 移动模型到cuda
mynet.eval()
loop = tqdm(enumerate(test_loader), total = len(test_loader))
test_loss = 0.0
test_corrects = 0
test_num = 0
test_loss_all = []
test_acc_all = []
for step, (b_x, b_y) in loop:
    b_x = b_x.cuda() if torch.cuda.is_available() else b_x
    b_y = b_y.cuda() if torch.cuda.is_available() else b_y
    mynet.eval()  # 3设置模型为评估模式
    output = mynet(b_x)
    pre_lab = torch.argmax(output, 1)
    loss = criterion(output, b_y)
    test_loss += loss.item() * b_x.size(0)
    test_corrects += torch.sum(pre_lab == b_y.data)
    test_num += b_x.size(0)

test_loss_all.append(test_loss / test_num)
test_acc_all.append(test_corrects.double().item() / test_num)
print('Test Loss:{:.4f}  Test Acc:{:.4f}'.format(test_loss_all[-1], test_acc_all[-1]))
# 100%|███████████████████████████████████████████████████████████████████████| 40/40 [00:03<00:00, 10.42it/s]
# Val Loss:0.2298  Val Acc:0.9197
