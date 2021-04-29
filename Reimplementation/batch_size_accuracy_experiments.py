from datasets import *
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

def train_model(seed, data, fname):
    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_train_samples = len(data)
    num_test_samples = len(data.test_examples)
    print("Training dataset has "+ str(num_train_samples) + " points.")
    print("Test set has " + str(num_test_samples) + " points.")

    # Specify model
    #model = CIFARNet().to(device)
    model = MNISTNet().to(device)

    # Specify hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5

    # Create dataloaders
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

    # Specify loss function
    criterion = nn.CrossEntropyLoss()

    # Specify optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # track losses and accuracies
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        train_loss_this_epoch = 0
        train_correct_this_epoch = 0
        test_loss_this_epoch = 0
        test_correct_this_epoch = 0

        start = time.time()
        # Training
        for train_samples, train_labels in train_loader:
            # Seperate into labels, features and indices
            labels = train_labels # (64,)
            features = train_samples[0] # (64, 28, 28)

            # Send to GPU if available
            labels, features = labels.to(device), features.to(device)

            # Basic update loop
            optimizer.zero_grad()
            out = model(features)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss_this_epoch += loss.cpu().detach().numpy()
            train_correct_this_epoch += calc_num_correct(out, labels)

        with torch.no_grad():
            test_examples = data.test_examples.to(device)
            test_labels = data.test_labels.to(device)

            # Compute output for test examples
            test_out = model(test_examples)

            # Compute loss
            test_loss = criterion(test_out, test_labels)
            test_loss_this_epoch += test_loss.cpu().detach().numpy()

            # Compute accuracy
            test_correct_this_epoch += calc_num_correct(test_out, test_labels)

        print("Epoch " + str(epoch) + ":")
        print("Train loss: " + str(train_loss_this_epoch / num_train_samples))
        print("Test loss: " + str(test_loss_this_epoch / num_test_samples))
        print("Train acc: " + str(train_correct_this_epoch / num_train_samples * 100) + "%")
        print("Test acc: " + str(test_correct_this_epoch / num_test_samples * 100) + "%")

        train_losses.append(train_loss_this_epoch / num_train_samples)
        train_accuracies.append(train_correct_this_epoch / num_train_samples)
        test_losses.append(test_loss_this_epoch / num_test_samples)
        test_accuracies.append(test_correct_this_epoch / num_test_samples)

    print("Max Train acc: " + str(max(train_accuracies)))
    print("at: " + str(np.argmax(train_accuracies)))
    print("Max Test acc: " + str(max(test_accuracies)))
    print("at: " + str(np.argmax(test_accuracies)))

    file = open("./output/batch" + fname + "_accuracies_seed" + str(seed) + ".txt", "a")
    file.write(str(test_accuracies[-1]) + "\n")
    file.close()

def calc_num_correct(pred, labels):
    pred, labels = pred.cpu(), labels.cpu()
    pred_argmax = torch.argmax(pred, dim=1)
    return torch.sum((torch.eq(pred_argmax, labels))).item()

if __name__ == "__main__":
    # Set seed (should be 1 - 5 according to paper)

    seeds = [2]
    dataset = "MNIST"  # "PMNIST" "CIFAR"
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        ks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        '''
        forgeting_events = "./forgetting_events/batch10%_forgetting_events_MNIST_seed1.pt"
        for k in ks:
            data = RemoveTopKMNISTDataset("./data/" + dataset + "/processed",
                                          forgetting_events_file=forgeting_events)
            data.remove_top_k_data(k=k)
            train_model(seed=seed, data=data, fname="10%")

        forgeting_events = "./forgetting_events/forgetting_events_MNIST_seed1.pt"
        for k in ks:
            data = RemoveTopKMNISTDataset("./data/" + dataset + "/processed",
                                          forgetting_events_file=forgeting_events)
            data.remove_top_k_data(k=k)
            train_model(seed=seed, data=data, fname="64")
        '''
        forgeting_events = "./forgetting_events/forgetting_events_MNIST_seed1.pt"
        for k in ks:
            data = RemoveTopKMNISTDataset("./data/" + dataset + "/processed",
                                          forgetting_events_file=forgeting_events)
            data.remove_random_k_data(k=k)
            train_model(seed=seed, data=data, fname="random")

