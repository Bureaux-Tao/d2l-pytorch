##
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils import data
from torchvision import transforms
import pandas as pd
import time
import copy
import seaborn as sns
import os
from tqdm import tqdm

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
plt.show()

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
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 定义一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 3), stride = (1, 1), padding = 1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2, stride = 2),
        )
        # 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), (1, 1), 0),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    # 定义网络的前向传播路径
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


# 输出我们的网络结构
myconvnet = ConvNet()
print(myconvnet)


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
    since = time.time()
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
    train_process = pd.DataFrame(
        data = {
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all
        }
    )
    return model, train_process


##
optimizer = torch.optim.Adam(myconvnet.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()  ##损失函数
myconvnet, train_process = train_model(
    myconvnet, train_loader, 0.8, criterion, optimizer, num_epochs = 30
)
# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)
print(train_process)

#     epoch  train_loss_all  val_loss_all  train_acc_all  val_acc_all
# 0       0        0.748373      0.549178       0.721208     0.792417
# 1       1        0.517185      0.478542       0.805937     0.822250
# 2       2        0.458603      0.435455       0.832167     0.840917
# 3       3        0.421484      0.406536       0.846521     0.852250
# 4       4        0.394251      0.386013       0.856896     0.858750
# 5       5        0.371970      0.369207       0.864792     0.866833
# 6       6        0.353533      0.356236       0.871250     0.870750
# 7       7        0.337942      0.345008       0.876333     0.875000
# 8       8        0.324731      0.335160       0.880833     0.878667
# 9       9        0.312977      0.327312       0.885000     0.882583
# 10     10        0.302196      0.319455       0.888896     0.886333
# 11     11        0.292416      0.310496       0.892229     0.888667
# 12     12        0.283555      0.304308       0.895625     0.890500
# 13     13        0.275155      0.297880       0.898771     0.894167
# 14     14        0.267477      0.293294       0.902250     0.895833
# 15     15        0.260270      0.288354       0.904771     0.897750
# 16     16        0.253687      0.284989       0.906667     0.899333
# 17     17        0.247129      0.282049       0.909188     0.900667
# 18     18        0.241246      0.278770       0.911042     0.901917
# 19     19        0.235494      0.275706       0.913417     0.903750
# 20     20        0.230146      0.273770       0.915063     0.904500
# 21     21        0.225122      0.271422       0.916958     0.905167
# 22     22        0.219788      0.269786       0.918646     0.905667
# 23     23        0.214997      0.268528       0.920792     0.906167
# 24     24        0.210313      0.268243       0.922479     0.905917
# 25     25        0.205782      0.267816       0.924083     0.906667
# 26     26        0.201326      0.266777       0.926042     0.907000
# 27     27        0.196946      0.266776       0.927646     0.907750
# 28     28        0.192727      0.266424       0.928937     0.908000
# 29     29        0.188566      0.266745       0.930500     0.907417

##
# 可视化模型训练过程
plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 400  #图片像素
plt.rcParams['figure.dpi'] = 400  #分辨率

plt.figure(figsize = (12, 4))
# 损失函数
plt.subplot(1, 2, 1)
plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label = "Train loss")
plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label = "Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
# 精度
plt.subplot(1, 2, 2)
plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label = "Train acc")
plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label = "Val acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()

##
# 对测试集进行预测，并可视化预测结果
myconvnet.eval()
output = myconvnet(test_data_x)
pre_lab = torch.argmax(output, 1)
acc = accuracy_score(test_data_y, pre_lab)
print("在测试集上的预测精度为：", acc)
# 在测试集上的预测精度为： 0.8897

# 计算混淆矩阵并可视化
conf_mat = confusion_matrix(test_data_y, pre_lab)
df_cm = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
heatmap = sns.heatmap(df_cm, annot = True, fmt = "d", cmap = "YlGnBu")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 45, ha = 'right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# https://blog.csdn.net/hhhhxxn/article/details/110734822
