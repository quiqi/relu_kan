import matplotlib.pyplot as plt
from torch_relu_kan import ReLUKANLayer, ReLUKAN
import torch
import numpy as np
from tqdm import tqdm


def gs(x, sigma = 5):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = (x - 0.5) * 2
    return np.multiply(np.power(np.sqrt(2 * np.pi) * sigma, -1), np.exp(-np.power(x, 2) / 2 * sigma ** 2))


#%% data
xs = np.arange(0, 5000) / 5000
ys = []
for i in range(5):
    x = xs[i*1000: (i+1)*1000]
    y = gs(x)
    ys.append(y)
ys = np.concatenate(ys)
plt.plot(xs, ys)
plt.show()

#%% 拟合
ys = ys / 0.08
xs = xs.reshape([5000, 1, 1])
ys = ys.reshape([5000, 1, 1])
xs = torch.Tensor(xs)
ys = torch.Tensor(ys)
relu_kan = ReLUKAN([1, 1], 25, 1)
opt = torch.optim.Adam(relu_kan.parameters())
mse = torch.nn.MSELoss()
plt.ion()
for i in range(5):
    t_xs = xs[i*1000: (i+1)*1000]
    t_ys = ys[i*1000: (i+1)*1000]
    for e in range(250):
        relu_kan.train()
        opt.zero_grad()
        t_pred = relu_kan(t_xs)
        loss = mse(t_pred, t_ys)
        loss.backward()
        opt.step()
        print(f'{i}:{e}')
        relu_kan.eval()
        pred = relu_kan(xs)
        plt.clf()
        plt.plot(xs[:, 0, 0], ys[:, 0, 0], '--', color='black')
        plt.plot(xs[:, 0, 0], pred.detach()[:, 0, 0], '-', color='black')
        if e == 249:
            plt.savefig(f'./data/cf_{i+1}.pdf', dpi=600)
        plt.pause(0.01)


#%% 绘制波形
for i in range(5):
    t_xs = xs[i*1000: (i+1)*1000]
    t_ys = ys[i*1000: (i+1)*1000]
    plt.clf()
    plt.plot(xs[:, 0, 0], ys[:, 0, 0], '--', color='black')
    plt.plot(t_xs[:, 0, 0], t_ys[:, 0, 0], '-', color='black')
    plt.savefig(f'./data/ps_{i + 1}.pdf', dpi=600)
