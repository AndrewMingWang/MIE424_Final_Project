import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
from os.path import isfile, join
from os import listdir

class MNISTDataset(Dataset):
    def __init__(self, processed_data_folder):
        # Load training data
        data = torch.load(processed_data_folder + "/training.pt")
        self.examples = data[0].float()
        self.labels = data[1].long()
        self.num_examples = self.examples.shape[0]

        # Load testing data
        test_data = torch.load(processed_data_folder + "/test.pt")
        self.test_examples = test_data[0].float()
        self.test_labels = test_data[1].long()
        self.num_test_examples = self.test_examples.shape[0]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return (self.examples[idx], idx), self.labels[idx]