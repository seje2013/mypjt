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
import post_training_quantization

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',        default='cifar100', help='cifar10|cifar100')
parser.add_argument('--model',          default='resnet18', help='resnet18|resnet34')
parser.add_argument('--load_path',      default='100epoch_77acc.pth', help='load weight file')
parser.add_argument('--quantize',       default=True, type=bool, help='Quantize model')
parser.add_argument('--qbit',           default=8, type=int, help='Quantization Precision')
parser.add_argument('--bn_fuse',        default=True, type=bool, help='fuse batch norm layer')
parser.add_argument('--batch_size',     default=64, type=int, help='input batch size for calibration (default: 64)')
parser.add_argument('--log_interval',   default=100, type=int, help='how many batches to wait before logging training status')
parser.add_argument('--distribution',   default=True, type=bool, help='visualize distribution')

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
    model = post_training_quantization.weight_ptq(model, args.qbit)

    activation = {}
    act_num = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            handle = module.register_forward_hook(utils.get_activation(activation, act_num))
            act_num += 1
    data, target = iter(test_loader).next()
    data = data.cuda()
    output = model(data)
    threshold = post_training_quantization.calibration(activation, act_num, args.qbit)
    model = post_training_quantization.activation_ptq(model, args.qbit, threshold)
    handle.remove()
    
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):            
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1)[1]
        total += target.size(0)
        correct += pred.eq(target).sum().item()
acc = 100 * correct / total
print(f"Test Top1 accuracy: {acc:.2f}")

if args.distribution:
    weight_num = 0
    weight = {}
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weight[weight_num] = module.weight.data
            weight_num += 1

    activation = {}
    act_num = 0
    for name, module in model.named_modules():
        if isinstance(module, post_training_quantization.QReLU):
            handle = module.register_forward_hook(utils.get_activation(activation, act_num))
            act_num += 1
    data, target = iter(test_loader).next()
    data = data.cuda()
    output = model(data)

    utils.plot_multi_layer_dist(weight, args.qbit, 'weight')
    utils.plot_multi_layer_dist(activation, args.qbit, 'activation')
    
    handle.remove()
