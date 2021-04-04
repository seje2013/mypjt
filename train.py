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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',        default='cifar100', help='cifar10|cifar100')
parser.add_argument('--model',          default='resnet18', help='resnet18|resnet34')
parser.add_argument('--quantize',       default=False, type=bool, help='Quantize INT8')
parser.add_argument('--batch_size',     default=64, type=int, help='input batch size for training (default: 64)')
parser.add_argument('--epochs',         default=100, type=int, help='number of epochs to train (default: 10)')
parser.add_argument('--log_interval',   default=100, type=int, help='how many batches to wait before logging training status')
parser.add_argument('--test_interval',  default=1,  type=int, help='how many epochs to wait before another test')
parser.add_argument('--lr',             default=0.01, type=float, help='learning rate (default: 1e-2)')
parser.add_argument('--tensorboard',    default=True, type=bool, help='using tensorboard')
parser.add_argument('--data_mixup',     default=True, type=bool, help='training data mixup augmentation')

current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.manual_seed(28)

print("==== initalization step ====")
print(args)
train_loader, test_loader = dataset.get_dataloader(args.dataset, args.batch_size)

model = model.get_model(args.model, num_classes=100)
model.cuda()

#state_dict = torch.load('/home/seje/study/mypjt/save_model/40epoch_74acc.pth')
#model.load_state_dict(state_dict['model'])

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=args.epochs)

if args.data_mixup:
    criterion = utils.CrossEntropyLoss()
else:
    criterion = nn.CrossEntropyLoss()

best_acc, old_file = 0, None

data_config ={
    'dataset' : 'CIFAR100',
    'n_classes': 100,
    'mixup_alpha': 1
    }

if args.tensorboard:
    writer = SummaryWriter(log_dir=f"/home/seje/study/mypjt/board/{current_time}")

for epoch in range(1, args.epochs+1):
    print("\n")
    print("==> Training step..")
    start_time = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.data_mixup == True:
            data, target = utils.mixup(data, target, data_config['mixup_alpha'], data_config['n_classes'])
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx % args.log_interval == 0:
            print(f"\r==> epoch:{epoch}[{batch_idx}/{len(train_loader)}] |loss:{loss.data:.3f}", end="")

    elapse_time = time.time() - start_time
    print(f"\nTraining Elapsed Time: {elapse_time:.2f}s/epoch")
    print(f"Train Epoch: [{epoch}/{args.epochs}] |Loss: {loss.data:.4f} |lr: {optimizer.param_groups[0]['lr']:.4f}")
    
    if epoch % args.test_interval == 0:
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
        print(f"Test Epoch: [{epoch}/{args.epochs}] |acc: {acc:.2f}")
        if acc > best_acc:
            print(f"Best Accuracy Renew: [{best_acc:.2f}->{acc:.2f}]")
            state = {
                'model': model.state_dict(),
                'args' : args
            }            
            new_file = os.path.join('/home/seje/study/mypjt/save_model', f'{epoch}epoch_{acc:.0f}acc.pth')
            torch.save(state, new_file)
            if old_file:
                os.remove(old_file)
            old_file = new_file
            best_acc = acc

    if args.tensorboard:
        writer.add_scalar("loss", loss.data, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("accuracy", acc, epoch)
        writer.flush()
