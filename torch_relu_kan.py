import numpy as np
import torch
import torch.nn as nn

class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int):
        super().__init__()
        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k+1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=False)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=False)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))
    def forward(self, x):
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_height - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.output_size, 1))
        return x


class ReLUKAN(nn.Module):
    def __init__(self, width, grid, k):
        super().__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLUKANLayer(width[i], grid, k, width[i+1]))
            # if len(width) - i > 2:
            #     self.rk_layers.append()
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        # x = x.reshape((len(x), self.width[-1]))
        return x


def show_base(phase_num, step):
    rk = ReLUKANLayer(1, phase_num, step, 1)
    x = torch.Tensor([np.arange(-600, 1024+600) / 1024]).T
    x1 = torch.relu(x - rk.phase_low)
    x2 = torch.relu(rk.phase_height - x)
    y = x1 * x1 * x2 * x2 * rk.r * rk.r
    for i in range(phase_num+step):
        plt.plot(x, y[:, i:i+1].detach(), color='black')
    plt.show()
    print('1')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    is_cuda = torch.cuda.is_available()
    show_base(5, 3)
    rk = ReLUKANLayer(1, 100, 5, 2)
    x = torch.Tensor([np.arange(0, 1024) / 1024]).T
    if is_cuda:
        rk.cuda()
        x = x.cuda()
    y = rk(x).detach().cpu()
    plt.show()

