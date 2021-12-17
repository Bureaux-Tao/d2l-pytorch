##
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
import os
from tqdm import tqdm

from d2l.earlystopping import EarlyStopping

os.environ["OMP_NUM_THREADS"] = "1"

##使用FasnionMNIST数据，准备训练数据集

train_data = torchvision.datasets.FashionMNIST(
    root = "/Users/Bureaux/Documents/workspace/PyCharmProjects/TorchProject/data",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
# 定义一个数据加载器
train_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = 16,
    shuffle = False,
    num_workers = 2,
)
# 计算train_loader有多少个batch
print("train_loader的batch数量为：", len(train_loader))
# train_loader的batch数量为： 938

##
# 获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
# 可视化一个batch的图像
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train_data.classes
class_label[0] = "T-shirt"
plt.figure(figsize = (12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap = plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size = 9)
    plt.axis("off")
    plt.subplots_adjust(wspace = 0.05)
# plt.show()

##
# 对测试集进行处理
test_data = torchvision.datasets.FashionMNIST(
    root = "/Users/Bureaux/Documents/workspace/PyCharmProjects/TorchProject/data",
    train = False,
    download = True
)
##为数据添加一个通道维度,并且取值范围缩放到0-1之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim = 1)
test_data_y = test_data.targets
print("test_data_x.shape:", test_data_x.shape)
print("test_data_y.shape:", test_data_y.shape)


# test_data_x.shape: torch.Size([10000, 1, 28, 28])
# test_data_y.shape: torch.Size([10000])


##卷积神经网络搭建
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义一个卷积层
        self.model = nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size = 5, padding = 2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(6, 16, kernel_size = 5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    # 定义网络的前向传播路径
    def forward(self, x):
        output = self.model(x)
        return output


# 输出我们的网络结构
lenet = LeNet()
print(lenet)


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
                                   path = '/Users/Bureaux/Documents/workspace/PyCharmProjects/TorchProject/weights/lenet.pt')
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
optimizer = torch.optim.Adam(lenet.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()  ##损失函数

##
mylenet, history = train_model(
    lenet, train_loader, 0.8, criterion, optimizer, num_epochs = 200
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
# myconvnet.load_state_dict(torch.load("weights/lenet.pt"))
mylenet.eval()
output = mylenet(test_data_x)
pre_lab = torch.argmax(output, 1)
acc = accuracy_score(test_data_y, pre_lab)
print("在测试集上的预测精度为：", acc)
# 在测试集上的预测精度为： 0.8984

##
# 计算混淆矩阵并可视化
conf_mat = confusion_matrix(test_data_y, pre_lab)
df_cm = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
heatmap = sns.heatmap(df_cm, annot = True, fmt = "d", cmap = "YlGnBu")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 45, ha = 'right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
