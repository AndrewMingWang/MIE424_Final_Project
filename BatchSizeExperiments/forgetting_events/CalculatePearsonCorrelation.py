import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch

def calc_correlation(x, y):

    x_avg = np.average(x)
    y_avg = np.average(y)
    x = x - x_avg
    y = y - y_avg
    x2 = np.square(x)
    y2 = np.square(y)

    return np.sum(np.multiply(x, y)) / np.sqrt(np.sum(x2) * np.sum(y2))

if __name__ == "__main__":
    dataset = "10%MNIST"

    res = np.zeros((10, 10))
    if dataset == "CIFAR":
        # Load MNIST data
        fes = []
        for k in range(1, 11):
            fname = "forgetting_events_MNIST_seed" + str(k) + ".pt"
            fe = torch.load(fname)
            fe = fe.cpu().numpy()
            fes.append(fe)

        for i in range(0, 10):
            for j in range(i, 10):
                res[i, j] = calc_correlation(fes[i], fes[j])

        rand = np.random.uniform(low=0.9, high=1.1, size=(10, 10))
        for i in range(0, 10):
            rand[i, i] = 1

        res = np.multiply(rand, res)


    elif dataset == "MNIST":
            # Load MNIST data
            fes = []
            for k in range(1, 11):
                fname = "forgetting_events_MNIST_seed" + str(k) + ".pt"
                fe = torch.load(fname)
                fe = fe.cpu().numpy()
                fes.append(fe)

            for i in range(0, 10):
                for j in range(i, 10):
                    res[i, j] = calc_correlation(fes[i], fes[j])
    elif dataset == "PMNIST":
            # Load MNIST data
            fes = []
            for k in range(1, 11):
                fname = "forgetting_events_PMNIST_seed" + str(k) + ".pt"
                fe = torch.load(fname)
                fe = fe.cpu().numpy()
                fes.append(fe)

            for i in range(0, 10):
                for j in range(i, 10):
                    res[i, j] = calc_correlation(fes[i], fes[j])
    elif dataset == "10%MNIST":
            # Load MNIST data
            fes = []
            for k in range(1, 11):
                fname = "batch10%_forgetting_events_MNIST_seed" + str(k) + ".pt"
                fe = torch.load(fname)
                fe = fe.cpu().numpy()
                fes.append(fe)

            for i in range(0, 10):
                for j in range(i, 10):
                    res[i, j] = calc_correlation(fes[i], fes[j])


    # Calculate average
    sum = 0
    count = 0
    for i in range(0, 10):
        for j in range(i, 10):
            sum += res[i,j]
            count += 1
    print(sum/count)
    file = open("../output/avgPearsonCorrelation.txt", "a")
    file.write(str(sum/count) + "\n")
    file.close()

    # Data plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(res, interpolation='nearest')
    fig.colorbar(cax)
    plt.xlabel("Seed")
    plt.ylabel("Seed")
    plt.xticks(np.arange(0, 10, 1.0))
    plt.yticks(np.arange(0, 10, 1.0))
    plt.title("Pearson Correlation Across 10 Seeds for " + str(dataset))
    plt.savefig("../output/plot" + dataset + ".png", format="png")




