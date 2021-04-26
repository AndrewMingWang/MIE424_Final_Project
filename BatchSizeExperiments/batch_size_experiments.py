from datasets import *
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

def train_model(seed, k):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Specify dataset
    dataset = "MNIST" # "PMNIST" "CIFAR"

    data = MNISTDataset("./data/" + dataset + "/processed")
    #data = CIFARDataset("./data/CIFAR/cifar-10-batches-py/output_data", "./data/CIFAR/cifar-10-batches-py/labels.txt")
    num_train_samples = len(data)
    num_test_samples = len(data.test_examples)
    print("Training dataset has "+ str(num_train_samples) + " points.")

    # Specify model
    #model = CIFARNet().to(device)
    model = MNISTNet().to(device)

    # Specify hyperparameters
    num_epochs = 200
    batch_size = int(float(k / 100) * num_train_samples)
    print("Batch Size: " + str(batch_size))
    learning_rate = 0.01
    momentum = 0.5

    # Create dataloaders
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

    # Specify loss function
    criterion = nn.CrossEntropyLoss()

    # Specify optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # track losses and accuracies
    train_losses = []
    train_accuracies = []

    # track forgetting events per example
    forgetting_events = torch.zeros(num_train_samples).to(device)

    # track the previous iterations classification, 0 implies incorrect, 1 implies correct
    last_classification = torch.zeros(num_train_samples).to(device)
    last_classification = torch.gt(last_classification, 0) # Convert 1-0 to True-False

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
            indices = train_samples[1] # (64,)

            # Send to GPU if available
            labels, features, indices = labels.to(device), features.to(device), indices.to(device)

            # Basic update loop
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            # Get current classification
            predictions = torch.argmax(predictions, dim=1) # (64, 10) -> (64,)
            curr = torch.eq(predictions, labels) # (64,)

            # Get last classification for the examples in this batch
            last = torch.gather(last_classification, dim=0, index=indices)

            # Check if there are any forgetting events
            forgot = torch.logical_and(last, torch.logical_not(curr)) # (64,)

            # Track number of forgetting events
            forgetting_events_this_batch = torch.gather(forgetting_events, dim=0, index=indices)
            forgetting_events_this_batch += forgot.long()
            forgetting_events.scatter_(dim=0, index=indices, src=forgetting_events_this_batch)

            # Update last classification
            last_classification.scatter_(dim=0, index=indices, src=curr)

            train_loss_this_epoch += loss.cpu().detach().numpy()
            train_correct_this_epoch += torch.sum(curr).item()

        # Test accuracy
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

        print("Epoch " + str(epoch) + " (" + (str(time.time() - start)[0:4]) + "s):")
        print("Train loss: " + str(train_loss_this_epoch / num_train_samples))
        print("Train acc: " + str(train_correct_this_epoch / num_train_samples * 100) + "%")
        print("Test acc: " + str(test_correct_this_epoch / num_test_samples * 100) + "%")
        print("Num forgetting events: " + str(torch.sum(forgetting_events.long()).item()))
        print("")

    # Saves a (60,000,) vector with the number of forgetting events per example
    torch.save(forgetting_events, "./batch" + str(k) + "%_forgetting_events_" + dataset + "_seed" + str(seed) + ".pt")

    # Saves final test and train accuracies
    file = open("./BatchSizeAccuracies.txt", "a")
    file.write(str(train_correct_this_epoch / num_train_samples * 100) + "\n")
    file.write(str(test_correct_this_epoch / num_test_samples * 100) + "\n")
    file.write("\n")
    file.close()


def calc_num_correct(pred, labels):
    pred, labels = pred.cpu(), labels.cpu()
    pred_argmax = torch.argmax(pred, dim=1)
    return torch.sum((torch.eq(pred_argmax, labels))).item()

if __name__ == "__main__":
    # Set seed (should be 1 - 5 according to paper)

    batch_size_percentages = [0.05]
    for k in batch_size_percentages:
        train_model(seed=1, k=k)
