##
import copy

import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

##

with open('../../data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()
print(corpus_chars[:40])
# 想要有直升机
# 想要和你飞到宇宙去
# 想要和你融化在一起
# 融化在宇宙里
# 我每天每天每

##
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')

##
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print(vocab_size)
# 2582

##
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)


# chars: 想要有直升机 想要和你飞到宇宙去 想要和
# indices: [2361, 1399, 384, 1796, 1414, 150, 1610, 2361, 1399, 1431, 2032, 507, 2018, 190, 236, 1248, 1610, 2361, 1399, 1431]

##
def data_iter_random(corpus_indices, batch_size, num_steps, device = None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    
    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype = torch.float32, device = device), torch.tensor(Y, dtype = torch.float32,
                                                                                    device = device)


my_seq = list(range(50))
for X, Y in data_iter_random(my_seq, batch_size = 3, num_steps = 5):
    print('X: ', X, '\nY:', Y, '\n')


# X:  tensor([[ 0.,  1.,  2.,  3.,  4.],
#         [15., 16., 17., 18., 19.],
#         [30., 31., 32., 33., 34.]], device='cuda:0')
# Y: tensor([[ 1.,  2.,  3.,  4.,  5.],
#         [16., 17., 18., 19., 20.],
#         [31., 32., 33., 34., 35.]], device='cuda:0')
# X:  tensor([[40., 41., 42., 43., 44.],
#         [ 5.,  6.,  7.,  8.,  9.],
#         [25., 26., 27., 28., 29.]], device='cuda:0')
# Y: tensor([[41., 42., 43., 44., 45.],
#         [ 6.,  7.,  8.,  9., 10.],
#         [26., 27., 28., 29., 30.]], device='cuda:0')
# X:  tensor([[35., 36., 37., 38., 39.],
#         [10., 11., 12., 13., 14.],
#         [20., 21., 22., 23., 24.]], device='cuda:0')
# Y: tensor([[36., 37., 38., 39., 40.],
#         [11., 12., 13., 14., 15.],
#         [21., 22., 23., 24., 25.]], device='cuda:0')

##
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype = torch.float32, device = device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


for X, Y in data_iter_consecutive(my_seq, batch_size = 2, num_steps = 6):
    print('X: ', X, '\nY:', Y, '\n')
# X:  tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
#         [25., 26., 27., 28., 29., 30.]], device='cuda:0')
# Y: tensor([[ 1.,  2.,  3.,  4.,  5.,  6.],
#         [26., 27., 28., 29., 30., 31.]], device='cuda:0')
# X:  tensor([[ 6.,  7.,  8.,  9., 10., 11.],
#         [31., 32., 33., 34., 35., 36.]], device='cuda:0')
# Y: tensor([[ 7.,  8.,  9., 10., 11., 12.],
#         [32., 33., 34., 35., 36., 37.]], device='cuda:0')
# X:  tensor([[12., 13., 14., 15., 16., 17.],
#         [37., 38., 39., 40., 41., 42.]], device='cuda:0')
# Y: tensor([[13., 14., 15., 16., 17., 18.],
#         [38., 39., 40., 41., 42., 43.]], device='cuda:0')
# X:  tensor([[18., 19., 20., 21., 22., 23.],
#         [43., 44., 45., 46., 47., 48.]], device='cuda:0')
# Y: tensor([[19., 20., 21., 22., 23., 24.],
#         [44., 45., 46., 47., 48., 49.]], device='cuda:0')

##
num_hiddens = 256
num_layers = 2
# rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens) # 已测试
rnn_layer = nn.LSTM(input_size = vocab_size, hidden_size = num_hiddens, num_layers = num_layers)
# 由于双向循环神经网络使用了过去的和未来的数据， 所以我们不能盲目地将这一语言模型应用于任何预测任务。

num_steps = 35
batch_size = 2
state = None
X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)


# RNN:
# torch.Size([35, 2, 256]) 1 torch.Size([2, 256])
# LSTM:
# torch.Size([35, 2, 256]) 2 torch.Size([1, 2, 256])

##
def one_hot(x, n_class, dtype = torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype = dtype, device = x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


x = torch.tensor([0, 2])
one_hot(x, vocab_size)


def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None
    
    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size)  # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


##
def predict_rnn(prefix, num_chars, model, vocab_size, device, idx_to_char,
                char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device = device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim = 1).item()))
    return ''.join([idx_to_char[i] for i in output])


model = RNNModel(rnn_layer, vocab_size).to(device)
predict_rnn('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)


# '分开篮篮篮篮篮篮篮篮篮篮'

##
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device = device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def train(model, num_hiddens, vocab_size, device,
          corpus_indices, idx_to_char, char_to_idx,
          num_epochs, num_steps, lr, clipping_theta,
          batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model.to(device)
    print(model)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            
            (output, state) = model(X, state)  # output: 形状为(num_steps * batch_size, vocab_size)
            
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())
            
            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))


##
num_epochs, batch_size, lr, clipping_theta = 500, 128, 1e-2, 1e-1  # 注意这里的学习率设置
pred_period, pred_len, prefixes = 50, 50, ['独自孤单', '夜太漫长']
train(model, num_hiddens, vocab_size, device,
      corpus_indices, idx_to_char, char_to_idx,
      num_epochs, num_steps, lr, clipping_theta,
      batch_size, pred_period, pred_len, prefixes)
