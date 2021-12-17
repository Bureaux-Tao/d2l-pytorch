##
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

boston = datasets.load_boston()
X = boston.data
y = boston.target
# X = X[y < 50.0]
# y = y[y < 50.0]
print(len(y) - len(y[y < 50.0]))

##
X_train, X_test, y_train, y_test = train_test_split(X, y)
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# (379, 13) (127, 13) (379,) (127,)

##
# net
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(n_feature, 16)
        # self.dropout = torch.nn.Dropout(0.2)
        # self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(16, n_output)
    
    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)
        # out = self.dropout(out)
        # out = self.linear2(out)
        # out = torch.relu(out)
        # out = self.dropout(out)
        out = self.linear3(out)
        return out


##
net = Net(13, 1)
# loss
loss_func = torch.nn.L1Loss()
# optimiter
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001, weight_decay = 0.01)
# training

train_loss_list = []
test_loss_list = []
for i in range(10000):
    x_data = torch.tensor(X_train, dtype = torch.float32)
    y_data = torch.tensor(y_train, dtype = torch.float32)
    # pred = net(x_data)
    pred = net.forward(x_data)
    # torch.Size([379, 1])
    # squeeze(a)就是将a中所有为1的维度删掉
    pred = torch.squeeze(pred)
    # torch.Size([379])
    loss = loss_func(pred, y_data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss_list.append(loss.item())
    if (i + 1) % 1000 == 0:
        print("epoch:{}, loss:{}".format(i + 1, loss))
    
    # test
    x_data = torch.tensor(X_test, dtype = torch.float32)
    y_data = torch.tensor(y_test, dtype = torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data)
    test_loss_list.append(loss_test.item())
    if (i + 1) % 1000 == 0:
        print("epoch:{}, val_loss:{}\n".format(i + 1, loss_test))

# torch.save(net, "boston_model.pkl")
# test

##
# net = torch.load("boston_model.pkl")
# loss_func = torch.nn.L1Loss()

x_data = torch.tensor(X_test, dtype = torch.float32)
y_data = torch.tensor(y_test, dtype = torch.float32)
pred = net.forward(x_data)
pred = torch.squeeze(pred)
loss_test = loss_func(pred, y_data)
print("loss_test:{}".format(loss_test))

# MSE:9.20052719116211
# MAE:1.9716278314590454

# print(train_loss_list)

##
# 可视化模型训练过程
plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # 图片像素
plt.rcParams['figure.dpi'] = 360  # 分辨率

plt.plot(range(1, 10001), train_loss_list, color = 'red', label = "Train loss")
plt.plot(range(1, 10001), test_loss_list, color = 'blue', label = "Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()
