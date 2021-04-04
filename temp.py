import argparse
import datetime
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import dataset
import model
import utils
import quantization

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',        default='cifar100', help='cifar10|cifar100')
parser.add_argument('--model',          default='resnet18', help='resnet18|resnet34')
parser.add_argument('--load_path',      default='100epoch_77acc.pth', help='load weight file')
parser.add_argument('--quantize',       default=True, type=bool, help='Quantize model')
parser.add_argument('--qbit',           default=8, type=int, help='Quantization Precision')
parser.add_argument('--bn_fuse',        default=True, type=bool, help='fuse batch norm layer')
parser.add_argument('--batch_size',     default=64, type=int, help='input batch size for training (default: 64)')
parser.add_argument('--log_interval',   default=100, type=int, help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.manual_seed(28)

print(args)
_ , test_loader = dataset.get_dataloader(args.dataset, args.batch_size)

model = model.get_model(args.model, num_classes=100)
model.cuda()

save_dir = '/home/seje/study/mypjt/save_model'
model_path = os.path.join(save_dir, args.load_path)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict['model'])



if args.bn_fuse:
    model = utils.fuse_bn_recursively(model)
if args.quantize:
    model = quantization.weight_ptq(model, args.qbit)
    activation = {}
    act_num = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            handle = module.register_forward_hook(utils.get_activation(activation, act_num))
            act_num += 1
    data, target = iter(test_loader).next()
    data = data.cuda()
    output = model(data)
    utils.plot_multi_layer_dist(activation, args.qbit)