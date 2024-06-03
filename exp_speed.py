import time
import matplotlib.pyplot as plt
from torch_relu_kan import ReLUKANLayer, ReLUKAN
import torch
import numpy as np


def train_loss_time(xs, ys, kan_module, epoch_max, cuda=True):
    if cuda:
        kan_module.cuda()
        xs = xs.cuda()
        ys = ys.cuda()

    opt = torch.optim.Adam(kan_module.parameters())
    mse = torch.nn.MSELoss()
    # 由于torch框架的懒加载问题，先进行一次前向传播后再统计训练时间
    pred = kan_module(xs)

    t = time.time()
    for e in range(epoch_max):
        opt.zero_grad()
        pred = kan_module(xs)
        loss = mse(pred[:, :, 0], ys)
        loss.backward()
        opt.step()

    return time.time() - t


#%% f1(x) = sin(pi*x)
relu_kan = ReLUKAN([1, 1], 5, 3)
xs = torch.Tensor([np.arange(0, 1000) / 1000]).T
ys = torch.sin(torch.pi * xs)
print('f1_CPU', train_loss_time(xs, ys, relu_kan, 500, cuda=False))
print('f1_GPU', train_loss_time(xs, ys, relu_kan, 500, cuda=True))

#%% f2(x) = sin(pi*x)
relu_kan = ReLUKAN([2, 1], 5, 3)
xs = np.random.random([1000, 2, 1])
ys = np.sin(np.pi * xs[:, 0, 0] + np.pi * xs[1, 0, 0])
ys.resize([1000, 1])
xs = torch.Tensor(xs)
ys = torch.Tensor(ys)
print('f1_CPU', train_loss_time(xs, ys, relu_kan, 500, cuda=False))
print('f1_GPU', train_loss_time(xs, ys, relu_kan, 500, cuda=True))

#%% f3(x) = arctan(x1 + x1*x2 + x2*x2)
relu_kan = ReLUKAN([2, 1, 1], 5, 3)
xs = np.random.random([1000, 2, 1])
ys = np.arctan(xs[:, 0, 0] + xs[:, 0, 0] * xs[:, 1, 0] + xs[:, 1, 0] * xs[:, 1, 0])
ys.resize([1000, 1])
xs = torch.Tensor(xs)
ys = torch.Tensor(ys)
print('f1_CPU', train_loss_time(xs, ys, relu_kan, 500, cuda=False))
print('f1_GPU', train_loss_time(xs, ys, relu_kan, 500, cuda=True))

#%% f3(x) = exp(sin(x1*x1 + x2*x2) + sin(x3*x3 + x4*x4))
relu_kan = ReLUKAN([4, 4, 2, 1], 10, 3)
xs = np.random.random([1000, 4, 1])
ys = np.exp(np.sin(xs[:, 0, 0] * xs[:, 0, 0] + xs[:, 1, 0] * xs[:, 1, 0]) +
            np.sin(xs[:, 2, 0] * xs[:, 2, 0] + xs[:, 3, 0] * xs[:, 3, 0]))

ys.resize([1000, 1])
xs = torch.Tensor(xs)
ys = torch.Tensor(ys)
print('f1_CPU', train_loss_time(xs, ys, relu_kan, 500, cuda=False))
print('f1_GPU', train_loss_time(xs, ys, relu_kan, 500, cuda=True))