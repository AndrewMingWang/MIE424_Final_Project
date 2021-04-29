import numpy as np
import torch
import matplotlib.pyplot as plt
import ast
import pandas as pd
import pickle
import glob
from PIL import Image


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from statistics import mean
import os
import matplotlib.cm as cm
import seaborn as sns
sns.set_theme(style="darkgrid")


#plot type #1 functions
def get_num_forget_events(x):
  return np.count_nonzero(np.diff(np.array(x)) == -1)
def get_data(dataset, seed, filename):
  with open(filename, 'rb') as f:
    data = pickle.load(f)
  num_forgets = []
  for key, value in data.items():
    if type(key) == np.int64:
      num_forgets.append(get_num_forget_events(value[1]))

  res = np.array([np.count_nonzero(np.array(num_forgets) == val) for val in range(0,max(num_forgets)+1)])/ (len(data)-2)
  
  return res

def plot(dataset,seed,filename):
  res = get_data(dataset,seed,filename)
  colors = {'mnist':'b','permuted_mnist': 'g', 'cifar10': 'r', 'cifar100': 'r'}
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
  plt.show()
  
  #600002, 60000 indices representing data points, then train and test which are just a list of accuracies of length num_iters *epochs
#data[i] is a list of 3 elements:
# element 0 represents the loss values over the 10 epochs it has been seen. element 1 is the accuracy (ie: either 0 or 1), element 2 is the margin (see paper for def)
def plot_2(non_noisy_res, noisy_res,dataset,seed):
  colors = {'mnist':'b','permuted_mnist': 'g', 'cifar10': 'r', 'cifar100': 'r'}
  fig = plt.figure(figsize=(14,9))
  ax = plt.axes()
  ax.bar([i for i in range(0,len(non_noisy_res))], non_noisy_res, width = 1, color = 'g', alpha =0.7)
  ax.bar([i for i in range(0,len(noisy_res))], noisy_res, width = 1, color = 'r',alpha =0.7)
  ax.set_ylim((0,max(max(non_noisy_res), max(noisy_res))+0.05))
  ax.set_xlim((0,max(len(non_noisy_res), len(noisy_res))))
  ax.set_xlabel('number of forgetting events')
  ax.set_ylabel('fraction of examples')
  plt.title(f"Histogram of Forgetting Events for {dataset} for seed = {seed}")
  plt.legend(['regular examples','noisy examples'])
  plt.xticks(visible=True)
  plt.yticks(visible=True)
  plt.show()
  
  
  
#plot type #2

def plot_exp2(changed_label_path, stats_dict_path,dataset,seed):
  filename = changed_label_path
  with open(filename) as f:
      noisy = f.readlines()
  noisy = [int(x.strip().split(' ')[0]) for x in noisy]
  non_noisy = [0 for i in range(0,40000)]
  noisy_num_forget = [0 for i in range(0,10000)]
  filename = stats_dict_path
  with open(filename, 'rb') as f:
    data = pickle.load(f)
  i = 0
  j = 0
  for key, value in data.items():
    if type(key) == np.int64 and key not in noisy:
      non_noisy[i] = get_num_forget_events(value[1])
      i+=1
    if type(key) == np.int64 and key in noisy:
      noisy_num_forget[j] = get_num_forget_events(value[1])
      j+=1
  non_noisy_res =  np.array([np.count_nonzero(np.array(non_noisy) == val) for val in range(0,max(non_noisy)+1)])/ (len(data)-len(noisy)-2)
  noisy_res =  np.array([np.count_nonzero(np.array(noisy_num_forget) == val) for val in range(0,max(noisy_num_forget)+1)])/ (len(noisy))
  plot_2(non_noisy_res, noisy_res,dataset,seed)
