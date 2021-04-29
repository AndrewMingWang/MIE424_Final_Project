import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from os.path import isfile, join
from os import listdir
from PIL import Image

class RemoveTopKMNISTDataset(Dataset):
    def __init__(self, processed_data_folder, forgetting_events_file="./forgetting_events_MNIST_seed1.pt"):
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

        # Load forgetting events
        self.forgetting_events = torch.load(forgetting_events_file)

        # Set to be used (after removing k% of the data)
        self.examples_use = self.examples
        self.labels_use = self.labels
        self.num_examples_use = self.num_examples

    def remove_top_k_data(self, k):
        ordering = torch.argsort(self.forgetting_events, descending=False)

        stop_idx = int(k * self.num_examples)
        to_be_kept = ordering[stop_idx:]

        self.examples_use = self.examples[to_be_kept]
        self.labels_use = self.labels[to_be_kept]
        self.num_examples_use = len(self.examples_use)

    def remove_random_k_data(self, k):
        stop_idx = int((1-k) * self.num_examples)

        permuted = torch.randperm(stop_idx)

        self.examples_use = self.examples[permuted]
        self.labels_use = self.labels[permuted]
        self.num_examples_use = len(self.examples_use)

    def __len__(self):
        return self.num_examples_use

    def __getitem__(self, idx):
        return (self.examples_use[idx], idx), self.labels_use[idx]

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

class CIFARDataset(Dataset):
    def __init__(self, data_folder, labels_file):
        self.labels_dict = {};

        f = open(labels_file, "r")
        labels_lines = f.read().splitlines()

        for line in labels_lines:
            label = int(line[-1])
            text = line[:-6]

            self.labels_dict[text] = label

        self.tensor_files = [join(data_folder, f) for f in listdir(data_folder)
                            if (isfile(join(data_folder, f))) and f[-3:] == ".pt"]
        self.tensor_indices = [os.path.basename(f)[:-3] for f in self.tensor_files]

        self.tensors = [torch.load(f) for f in self.tensor_files]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        label_idx = self.tensor_indices[idx]
        tensor = self.tensors[idx]
        label = self.labels_dict[label_idx]

        return (tensor.float(), idx), label


