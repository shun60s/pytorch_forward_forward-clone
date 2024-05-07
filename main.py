#coding:utf-8

"""
This is a copy from  [pytorch_forward_forward](https://github.com/mpezeshki/pytorch_forward_forward) main.py and some changes.
Please see LICENSE_pytorch_forward_forward. 

"""
"""
This is CPU  beside original uses CUDA.
Check version:
    Python 3.6.4 on win32
    torch  1.7.1+cpu
    torchvision 0.8.2+cpu
    tqdm  4.28.1
    matplotlib 3.3.1
""" 

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import argparse # add
import sys      # add


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    # change from the first 10 pixels to bottom line 10 pixels
    x_[:, 28*27 : 28*27+10] *= 0.0
    x_[range(x.shape[0]), 28*27 + y] = x.max()  # # x.max() is 2.8215
    return x_


class Net(torch.nn.Module):

    def __init__(self, dims, epochs, threshold ):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], epochs, threshold)]  #.cuda()]  # change

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,epochs, threshold, # change
                 bias=True):  # , device=None, dtype=None):  # change
        super().__init__(in_features, out_features, bias)  # , device, dtype) # change
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = threshold  # 2.0   # change
        self.num_epochs = epochs    # 1000  # change

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)  # # normalize
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
  
""" add start """
def visualize_sample2(data, name, idx=0):
    fig=plt.figure()
    for i in range(len(data)):  # len(data) must be 3.
        plt.subplot(int(131+i))
        plt.title(name[i])
        plt.imshow(data[i][idx].cpu().reshape(28, 28), cmap="gray")
        plt.minorticks_on()
        #plt.grid()
    fig.tight_layout()
    plt.show()
""" add end """
    
    
if __name__ == "__main__":
    """ add start """
    parser = argparse.ArgumentParser(description='pytorch forward forward ')
    parser.add_argument('--train-batch-size', type=int, default=50000, metavar='N',
                        help='input batch size for training (default: 50000)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--threshold', type=float, default=2.0, metavar='TH',
                        help='threshold  (default: 2.0)')
    parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    """ end of add """
    
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders(args.train_batch_size, args.test_batch_size) # change

    # # input dimension 784 is 28x28, MNIST data size.
    net = Net([784, 500, 500], args.epochs, args.threshold ).to(device) # change
    x, y = next(iter(train_loader))
    
    if use_cuda:      # add
        x, y = x.cuda(), y.cuda()
        
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))  # # x.size(0) is train_batch_size
    # change where y==y[rnd], increase one as label number. 10 becomes 0.
    x_neg = overlay_y_on_x(x, torch.where( torch.where(y == y[rnd], y+1, y[rnd]) == 10, 0, torch.where(y == y[rnd], y+1, y[rnd])) )
    
    """ change start """
    #for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    #    visualize_sample(data, name)
    # show two samples.
    visualize_sample2([x, x_pos, x_neg], ['orig', 'pos', 'neg'], idx=0) 
    visualize_sample2([x, x_pos, x_neg], ['orig', 'pos', 'neg'], idx=4) 
    """ change end """
    
    net.train(x_pos, x_neg)
    
    print('train error:[%]', (1.0 - net.predict(x).eq(y).float().mean().item()) * 100 )  # change
    
    x_te, y_te = next(iter(test_loader))
    if use_cuda:      # add
        x_te, y_te = x_te.cuda(), y_te.cuda()
    
    print('test error:[%]', (1.0 - net.predict(x_te).eq(y_te).float().mean().item()) * 100)  # change
    
    if args.save_model:   # add
        torch.save(net.state_dict(), "mnist_forward_forward.pt")  # add
