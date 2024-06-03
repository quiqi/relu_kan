import pickle

import matplotlib.pyplot as plt
from torch_relu_kan import ReLUKANLayer, ReLUKAN
import torch
import numpy as np
from tqdm import tqdm


# %% 测试类
class Train:
    def __init__(self, f, width, g, k):
        self.cuda = torch.cuda.is_available()
        self.relu_kan = ReLUKAN(width, g, k)
        self.f = f

        self.input_size = width[0]
        self.train_xs = np.random.random([1000, self.input_size, 1])
        self.train_ys = f(self.train_xs)
        self.test_xs = np.random.random([100, self.input_size, 1])
        self.test_ys = f(self.test_xs)

        self.train_xs = torch.Tensor(self.train_xs)
        self.train_ys = torch.Tensor(self.train_ys)
        self.test_xs = torch.Tensor(self.test_xs)
        self.test_ys = torch.Tensor(self.test_ys)

        self.train_loss = []
        self.test_loss = []

        if self.cuda:
            self.train_xs = self.train_xs.cuda()
            self.train_ys = self.train_ys.cuda()
            self.test_xs = self.test_xs.cuda()
            self.test_ys = self.test_ys.cuda()
            self.relu_kan.cuda()

        self.opt = torch.optim.Adam(self.relu_kan.parameters())
        self.loss_fun = torch.nn.MSELoss()

    def train_process(self, epoch_max: int = 1000):
        for e in tqdm(range(epoch_max)):
            self.train()
            self.test()

    def train(self):
        self.relu_kan.train()
        self.opt.zero_grad()
        pred = self.relu_kan(self.train_xs)
        loss = self.loss_fun(pred, self.train_ys)
        loss.backward()
        self.opt.step()
        self.train_loss.append(loss.item())

    def test(self):
        self.relu_kan.eval()
        pred = self.relu_kan(self.test_xs)
        loss = self.loss_fun(pred, self.test_ys)
        self.test_loss.append(loss.item())

    def plt_fitting(self, name, mode=1):
        plt.title(f'${name}$ effect')
        if self.input_size == 1 and mode==1:
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            xs = np.array([np.arange(0, 1000) / 1000]).T
            ys = self.f(xs)
            plt.plot(xs, ys, '--', color='black', label='true')
            xs = torch.Tensor(xs)
            if self.cuda:
                xs = xs.cuda()
            pred = self.relu_kan(xs)
            plt.plot(xs.cpu(), pred[:, :, 0].detach().cpu(), '-', color='black', label='pred')
            plt.legend()
            plt.show()
        else:
            plt.xlabel('pred')
            plt.ylabel('true')
            pred = self.relu_kan(self.test_xs)
            plt.plot(pred.detach().cpu().flatten(), self.test_ys.cpu().flatten(), '.', color='black')
        plt.savefig(f'./data/effect_{name}.pdf', dpi=600)
        plt.clf()

    def plt_loss(self, name: str):
        plt.title(f'${name}$ training process')
        plt.xlabel('iterations')
        plt.ylabel('MSE loss')
        plt.semilogy(self.train_loss, '-', color='black', label='train')
        plt.semilogy(self.test_loss, '--', color='black', label='test')
        plt.legend()
        plt.savefig(f'./data/process_{name}.pdf', dpi=600)
        plt.clf()

    def save_process(self, name):
        with open(f'./data/loss_{name}.pkg', 'wb') as f:
            pickle.dump({'train_loss': self.train_loss, 'test_losss': self.test_loss}, f)


# %% f1 = sin(pi * x)
def f1(x):
    return np.sin(np.pi * x)


def f2(x):
    return np.exp(x)


def f3(x):
    return x * x + x + 1


def f4(x):
    y = np.sin(np.pi * x[:, [0]] + np.pi * x[:, [1]])
    return y


def f5(x):
    y = np.exp(np.sin(np.pi * x[:, [0]]) + x[:, [1]] * x[:, [1]])
    return y


def f6(x):
    y = np.exp(
        np.sin(np.pi * x[:, [0]] * x[:, [0]] + np.pi * x[:, [1]] * x[:, [1]]) +
        np.sin(np.pi * x[:, [2]] * x[:, [2]] + np.pi * x[:, [3]] * x[:, [3]])
    )
    return y

def f7(x):
    return np.sin(5 * np.pi * x) + x


train_plan = {
    'f_1': (f1, [1, 1], 5, 3),
    # 'f_2': (f2, [1, 1], 5, 3),
    'f_3': (f3, [1, 1], 5, 3),
    'f_4': (f4, [2, 5, 1], 5, 3),
    'f_5': (f5, [2, 5, 1], 5, 3),
    'f_6': (f6, [4, 2, 2, 1], 5, 3),
    'f_2': (f7, [1, 1], 5, 3),
}

for f_name in train_plan:
    train = Train(*train_plan[f_name])
    train.train_process(5000)
    train.plt_loss(f_name)
    train.plt_fitting(f_name)
    train.save_process(f_name)
