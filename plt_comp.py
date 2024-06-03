import pickle
import matplotlib.pyplot as plt

for i in range(1, 7):
    with open(f'./data/f_{i}.pkg', 'rb') as f:
        kan = pickle.load(f)
    with open(f'./data/loss_f_{i}.pkg', 'rb') as f:
        r_kan = pickle.load(f)

    name = f'f_{i}'
    plt.title(f'${name}$ training process')
    plt.xlabel('iterations')
    plt.ylabel('MSE loss')
    plt.semilogy(kan['test_loss'], '--', color='black', label='KAN test loss')
    plt.semilogy(r_kan['test_losss'], '-', color='black', label='ReLU-KAN test loss')
    plt.legend()
    plt.savefig(f'./data/comp_process_{name}.pdf', dpi=600)
    plt.clf()