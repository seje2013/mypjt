from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_cifar10(batch_size, num_workers=4):
    data_root='/home/seje/study/mypjt/dataset/cifar10'
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    train_loader = DataLoader(
        datasets.CIFAR10(
            root=data_root, train=True, download=False,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        ), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        datasets.CIFAR100(
            root=data_root, train=False, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        ), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def get_cifar100(batch_size, num_workers=4):
    data_root='/home/seje/study/mypjt/dataset/cifar100'
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    train_loader = DataLoader(
        datasets.CIFAR100(
            root=data_root, train=True, download=False,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        ), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        datasets.CIFAR100(
            root=data_root, train=False, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        ), batch_size=batch_size, shuffle=False,
    )
    return train_loader, test_loader


def get_dataloader(dataset, batch_size, num_workers=4):
    if dataset == 'cifar10':
        train_loader, test_loader = get_cifar10(batch_size, num_workers)
    elif dataset == 'cifar100':
        train_loader, test_loader = get_cifar100(batch_size, num_workers)
    return train_loader, test_loader