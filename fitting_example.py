import matplotlib.pyplot as plt
from torch_relu_kan import ReLUKANLayer, ReLUKAN
import torch
import numpy as np


if __name__ == '__main__':
    # skan = ReLUKANLayer(1, 100, 20, 1)
    relu_kan = ReLUKAN([1, 1], 5, 3)
    x = torch.Tensor([np.arange(0, 1024) / 1024]).T
    y = torch.sin(5*torch.pi*x)
    if torch.cuda.is_available():
        relu_kan = relu_kan.cuda()
        x = x.cuda()
        y = y.cuda()

    opt = torch.optim.Adam(relu_kan.parameters())
    mse = torch.nn.MSELoss()

    plt.ion()
    losses = []
    for e in range(5000):
        opt.zero_grad()
        pred = relu_kan(x)
        loss = mse(pred[:, :, 0], y)
        loss.backward()
        opt.step()
    # print(time.time() - t)
        pred = pred.detach()
        plt.clf()
        plt.plot(x.cpu(), y.cpu())
        plt.plot(x.cpu(), pred[:, :, 0].cpu())
        plt.pause(0.01)
        print(loss)



