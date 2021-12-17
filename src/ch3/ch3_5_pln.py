##
import torch
from sklearn.metrics import accuracy_score
from torch import nn
import torchvision
from pytorch_lightning import LightningModule, Trainer, seed_everything, callbacks
from torch.utils import data
from torchvision import transforms
import os

os.environ["OMP_NUM_THREADS"] = "1"
BATCH_SIZE = 32

##使用FasnionMNIST数据，准备训练数据集

train_data = torchvision.datasets.FashionMNIST(
    root = "/Users/Bureaux/Documents/workspace/PyCharmProjects/TorchProject/data",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

# train_x = train_data[0:int(len(train_data) * 0.8)]
# val_x = train_data[int(len(train_data) * 0.8):]
train_len = int(len(train_data) * 0.8)
val_len = int(len(train_data) * 0.2)
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_len, val_len])

# 定义一个数据加载器
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 4,
)

val_loader = torch.utils.data.DataLoader(
    dataset = val_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 4,
)

# 计算train_loader有多少个batch
# print("train_loader的batch数量为：", len(train_loader))
print("train_loader的batch数量为：：", len(train_loader))
print("val_loader的batch数量为：", len(val_loader))

##
# 对测试集进行处理
test_data = torchvision.datasets.FashionMNIST(
    root = "/Users/Bureaux/Documents/workspace/PyCharmProjects/TorchProject/data",
    train = False,
    download = True,
    transform = transforms.ToTensor(),  # 测试集也要加
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 4,
)

# 为数据添加一个通道维度,并且取值范围缩放到0-1之间
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim = 1)
test_data_y = test_data.targets
print("test_data_x.shape:", test_data_x.shape)
print("test_data_y.shape:", test_data_y.shape)


##
class Model(LightningModule):
    def __init__(self):
        super().__init__()
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
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output
    
    # 定义loss,以及可选的各种metrics
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = nn.CrossEntropyLoss()(prediction, y)
        return loss
    
    # 定义optimizer,以及可选的lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)
        return {"optimizer": optimizer}
    
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return {"test_loss": loss}




    ##
seed_everything(1234)
model = Model()

ckpt_callback = callbacks.ModelCheckpoint(
    monitor = 'val_loss',
    save_top_k = 1,
    mode = 'min'
)

# gpus=0 则使用cpu训练，gpus=1则使用1个gpu训练，gpus=2则使用2个gpu训练，gpus=-1则使用所有gpu训练，
# gpus=[0,1]则指定使用0号和1号gpu训练， gpus="0,1,2,3"则使用0,1,2,3号gpu训练
# tpus=1 则使用1个tpu训练

trainer = Trainer(max_epochs = 30, gpus = 0, callbacks = [ckpt_callback], auto_lr_find = True)

##
# 断点续训
# trainer = pl.Trainer(resume_from_checkpoint='./lightning_logs/version_31/checkpoints/epoch=02-val_loss=0.05.ckpt')

trainer.fit(model, train_loader, val_loader)

##
# 使用模型
data_pred, label = next(iter(test_loader))
model.eval()
prediction = model(data_pred)
print(prediction)

##
model.eval()
output = model(test_data_x)
pre_lab = torch.argmax(output, 1)
acc = accuracy_score(test_data_y, pre_lab)
print("在测试集上的预测精度为：", acc)

##
trainer.save_checkpoint(
    '/Users/Bureaux/Documents/workspace/PyCharmProjects/TorchProject/weights/FashionMinist_pln.ckpt')

##
model_clone = Model.load_from_checkpoint(
    "/Users/Bureaux/Documents/workspace/PyCharmProjects/TorchProject/weights/FashionMinist_pln.ckpt")
# trainer_clone = Trainer()
model_clone.eval()
output = model_clone(test_data_x)
pre_lab = torch.argmax(output, 1)
acc = accuracy_score(test_data_y, pre_lab)
print("在测试集上的预测精度为：", acc)


# 在测试集上的预测精度为： 0.8875