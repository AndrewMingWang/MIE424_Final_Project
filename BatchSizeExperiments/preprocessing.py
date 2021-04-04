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

# Setup transforms for MNIST data
all_transforms = [
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, )) # I assume these must be known constants
]
transform = transforms.Compose(all_transforms)

# Download MNIST data into the ./data folder
trainset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)