from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import math

def plot_hist(input):
    input = input.clone().cpu().detach().flatten().numpy()
    min = input.min()
    max = input.max()
    mean = input.mean()
    std = input.std()
    fig = plt.figure()
    plt.hist(input, rwidth=0.5)
    plt.title(f'min:{min:.2f} | max:{max:.2f} | mean:{mean:.2f} | std:{std:.2f}')
    plt.show()

def plot_multi_layer_dist(input_dict, qbit, title):
    x_size = 4
    y_size = math.ceil(len(input_dict)/x_size)
    fig, axes = plt.subplots(x_size, y_size)
    fig.set_size_inches((10,6))
    fig.tight_layout()
    num = 0
    for i in range(x_size):
        for j in range(y_size):
            if i*y_size+j < len(input_dict):            
                data = input_dict[num]
                data = data.cpu().flatten().numpy()
                axes[i][j].hist(data, bins=int(2**qbit), rwidth=1, log=True)
                axes[i][j].set_title(f'{title}: {num}')
                num += 1
            else:
                axes[i][j].set_title('no layer')
    plt.show()
    return

def image_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)

def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]
    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)
    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)
    return data, targets

def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss

class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average
    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)


def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        m = block1
        conv = block2
        
        bn_st_dict = m.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = m.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv.weight.data.copy_(W)

        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias)
        else:
            conv.bias.data.copy_(bias)
            
        return conv
        
    else:
        return False
    
def fuse_bn_recursively(model):
    previous_name = None
    
    for module_name in model._modules:
        previous_name = module_name if previous_name is None else previous_name # Initialization
        conv_fused = fuse_single_conv_bn_pair(model._modules[module_name], model._modules[previous_name])
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()
            
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
            
        previous_name = module_name

    return model

def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
