# 《手动深度学习》书中代码

## Project Structure

```
./
├── README.md
├── __pycache__
├── d2l                            d2l工具包（可pip install）
│   ├── __init__.py
│   ├── __pycache__
│   ├── earlystopping.py           早停工具
│   ├── mxnet.py
│   ├── tensorflow.py
│   └── torch.py
├── data                           数据集
│   ├── FashionMNIST               FashionMNIST数据集
│   │   └── raw
│   ├── fra-eng                    英法翻译数据集
│   │   └── fra.txt
│   ├── jaychou_lyrics.txt         周杰伦歌词数据集
│   └── timemachine.txt            时光机文本数据集
├── environment.yaml               conda环境配置文件
├── lightning_logs
├── main.py
├── models
├── path.py
├── report                         nohup训练日志
│   ├── nohup_ch3_5.out
│   ├── nohup_ch6_6.out
│   ├── nohup_ch6_6_alt.out
│   ├── nohup_ch7_1.out
│   ├── nohup_ch7_1_alt.out
│   ├── nohup_ch7_2.out
│   ├── nohup_ch7_3.out
│   ├── nohup_ch7_3_alt.out
│   ├── nohup_ch7_4.out
│   ├── nohup_ch7_5.out
│   └── nohup_ch7_6.out
├── requirements.txt               pip包
├── src
│   ├── __init__.py
│   ├── ch2
│   │   ├── __init__.py
│   │   ├── ch2_1.py               数据操作
│   │   ├── ch2_3.py               线性代数
│   │   ├── ch2_5.py               自动微分
│   │   └── ch2_7.py               查阅文档
│   ├── ch3
│   │   ├── __init__.py
│   │   ├── ch3_2.py               线性回归的从零开始实现
│   │   ├── ch3_3.py               线性回归的简洁实现
│   │   ├── ch3_5.py               图像分类
│   │   ├── ch3_5_pln.py           图像分类(pytorch-lightning)
│   ├── ch4
│   │   ├── __init__.py
│   │   ├── ch4_4.py               模型选择、欠拟合和过拟合
│   │   ├── ch4_5.py               权重衰减
│   │   ├── ch4_6.py               Dropout
│   │   └── ch4_7.py               前向传播、反向传播和计算图
│   ├── ch5
│   │   ├── __init__.py
│   │   ├── ch5_1.py               层和块
│   │   ├── ch5_2.py               参数管理
│   │   ├── ch5_3.py               延后初始化
│   │   └── ch5_4.py               自定义层
│   ├── ch6
│   │   ├── __init__.py
│   │   ├── ch6_2.py               图像卷积
│   │   ├── ch6_3.py               填充和步幅
│   │   ├── ch6_5.py               汇聚层
│   │   ├── ch6_6.py               LeNet
│   │   └── ch6_6_alt.py           LeNet封装成类写法
│   ├── ch7
│   │   ├── __init__.py
│   │   ├── ch7_1.py               AlexNet书中写法
│   │   ├── ch7_1_alt.py           AlexNet调参
│   │   ├── ch7_2.py               VGG11
│   │   ├── ch7_3.py               NiN1*1卷积书中写法
│   │   ├── ch7_3_alt.py           NiN1*1封装训练函数
│   │   ├── ch7_4.py               Inception模块
│   │   ├── ch7_5.py               BatchNorm
│   │   └── ch7_6.py               残差网络
│   ├── ch8
│   │   ├── __init__.py
│   │   ├── ch8_2.py               文本预处理
│   │   ├── ch8_3.py               语言模型和数据集
│   │   ├── ch8_6.py               RNN/GRU/LSTM
│   │   └── ch8_6_alt.py           周杰伦歌词生成
│   ├── ch10
│   │   ├── __init__.py
│   │   ├── ch10_1.py              注意力提示
│   │   ├── ch10_3.py              注意力得分
│   │   └── ch10_4.py              Bahdanau注意力
│   └── ch9
│       ├── __init__.py
│       ├── ch9_5.py                     编码解码器
│       └── ch9_7.py                     英法互译任务
└── weights                                    保存的权重
    ├── FashionMinist_pln.ckpt
    ├── Inception.pt
    ├── alex_net.pt
    ├── batchnorm_alexnet.pt
    ├── lenet.pt
    ├── mlp.params
    ├── mydict
    ├── nin_net_adam.pt
    ├── nin_net_alt.pt
    ├── nin_net_sgd.pt
    ├── resnet.pt
    ├── vgg11_net.pt
    ├── x-file
    └── x-files

36 directories, 132 files
```

## To be continue...
...

...

...