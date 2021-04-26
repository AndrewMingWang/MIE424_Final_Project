import numpy as np
import torch
import matplotlib.pyplot as plt
import ast
import pandas as pd
import pickle
import glob
from PIL import Image
from matplotlib.ticker import PercentFormatter

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from statistics import mean
import os
import matplotlib.cm as cm
import seaborn as sns
sns.set_theme(style="darkgrid")

def plot(dataset, seed, big_vector):
    res = np.array([np.count_nonzero(np.array(big_vector) == val) for val in range(0,max(big_vector)+1)])/ (55000)
    colors = {'mnist':'b','permuted_mnist': 'g', 'cifar10': 'r', 'cifar100': 'r', '10%mnist' :'purple'}
    fig = plt.figure(figsize=(8,5))
    ax = plt.axes()
    ax.bar([i for i in range(0,len(res))], res, width = 1, color = colors[dataset])
    ax.set_ylim((0,1.1))
    ax.set_xlim((0,len(res)))
    ax.set_xlabel('number of forgetting events')
    ax.set_ylabel('fraction of examples')
    plt.title(f"Histogram of Forgetting Events for {dataset} for seed = {seed}")
    x1 = 0
    x2 = 10
    # select y-range for zoomed region
    y1 = 0
    y2 = max(res[1:])+0.05
    axes_kwargs = {'frame_on': True, 'alpha': 1}
    axins = inset_axes(ax,2.5,2.5, loc='upper right', borderpad =3, axes_kwargs=axes_kwargs) # zoom = 2
    axins.yaxis.get_major_locator().set_params(nbins=4)
    axins.xaxis.get_major_locator().set_params(nbins=4)
    plt.setp(axins.get_xticklabels(), visible=True)
    plt.setp(axins.get_yticklabels(), visible= True)
    axins.bar([i for i in range(0,len(res))], res, color = colors[dataset])
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=True)
    plt.yticks(visible=True)
    plt.draw()
    #plt.show()
    plt.savefig("./figures/" + dataset + str(seed) + ".png", format="png")

if __name__ == "__main__":

    for s in range(1, 11):
        fname = "./forgetting_events/forgetting_events_PMNIST_seed" + str(s) + ".pt"
        big = torch.load(fname).long()

        plot("permuted_mnist", seed=s, big_vector=big.cpu())

    for s in range(1, 11):
        fname = "./forgetting_events/forgetting_events_MNIST_seed" + str(s) + ".pt"
        big = torch.load(fname).long()

        plot("mnist", seed=s, big_vector=big.cpu())

    for s in range(1, 11):
        fname = "./forgetting_events/batch10%_forgetting_events_MNIST_seed" + str(s) + ".pt"
        big = torch.load(fname).long()

        plot("10%mnist", seed=s, big_vector=big.cpu().numpy())
