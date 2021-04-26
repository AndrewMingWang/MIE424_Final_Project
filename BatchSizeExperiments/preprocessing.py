import numpy as np
import numpy.random as npr
import time
import os
import sys
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# TO USE THIS ADD A "data" folder to the "BatchSizeExperiments" folder
# Then run this file

normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_dataset = datasets.CIFAR10(
    root='./data/CIFAR',
    train=True,
    transform=normalize,
    download=True)

test_dataset = datasets.CIFAR10(
    root='./data/CIFAR',
    train=False,
    transform=normalize,
    download=True)



'''
permuted = True
# Setup transforms for MNIST data
all_transforms = [
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, )) # I assume these must be known constants
]
if permuted:
    pixel_permutation = torch.randperm(28 * 28)
    all_transforms.append(
        transforms.Lambda(
            lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28)))
transform = transforms.Compose(all_transforms)

# Download MNIST data into the ./data folder
trainset = datasets.MNIST(
    root='./data2', train=True, download=True, transform=transform)
testset = datasets.MNIST(
    root='./data2', train=False, download=True, transform=transform)
'''