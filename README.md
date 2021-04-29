# MIE424 Final Project: Example Forgetting 

This repository contains code used to run the experiments for this project. It is based on the paper [An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://arxiv.org/abs/1812.05159) by Toneva et al. Most of the code is taken from the corresponding [repository](https://github.com/mtoneva/example_forgetting) with some changes made for exploratory experiments. 

## Code Requirements 

The libraries required are listed in requirements.txt. They can be installed using: 
```
pip install -r requirements.txt
```

## Experiments 

Our experiments all require the calculation of forgetting statistics. This is done using the following instructions: 

##### MNIST and permuted MNIST:
```
python run_mnist.py 
    --dataset [mnist/permuted_mnist]
    --no_dropout 
    --output_dir [mnist/permuted_mnist]_results
    --seed s
    --num_forget_batches n 
```
where s is the seed ranging from 1 to 5, inclusive, and n is the number of mini-batches sampled per update for forgetting. If running the original experiments from the paper, then n=1. The default for n is 1. Use options `--dataset mnist` and `--output_dir mnist_results` to run on MNIST, and options `--dataset permuted_mnist` and `--output_dir permuted_mnist_results` for permuted MNIST. Each training run with a different seed saves a file that contains the presentation statistics (loss, accuracy, misclassification margin) from that run in the specified `--output_dir`. The names of the saved files contain the arguments (and argument values) that were used to generate them.

##### CIFAR-10 and CIFAR-100:
```
python run_cifar.py 
    --dataset [cifar10/cifar100] 
    --data_augmentation 
    --output_dir [cifar10/cifar100]_results
    --seed s
    --num_forget_batches n 
```
where s ranges from to 1 to 5. The default setting was used for all other flags. This script has a similar functionality to `run_mnist.py`.

The above two scripts calculates the forgetting events and outputs each run as a pickle file within output_dir. This file can then be taken for plotting. 


In the following sections, we provide instructions to obtain the results from each of the experiments detailed in our report. They are organized by corresponding section number in the report. 

### 5.1.1 Distribution of Forgetting Events
Run a chosen dataset and seed, run the corresponding `run_{mnist/cifar}.py` file. With the resulting pickle file saved under path `pickle_path`, you would call `plot(dataset, seed, pickle_path)` where dataset is the name of the dataset (MNIST,PMNIST or CIFAR10) and seed is the seed used for the experiment.

### 5.1.2 Characteristics of Forgetful Examples
```
python run_cifar.py 
    --dataset cifar10 
    --data_augmentation 
    --output_dir cifar10_results 
    --noise_percent_labels 20
```

The resulting files you will get are `...changed_labels.txt` and `_stats_dict.pkl`. To get the same plot as verifcation experiment #2, you would call `plot_exp2(<path_to_changed_labels.txt>, <path_to_stats_dict.pkl>, pickle_path)` where dataset is the name of the dataset (MNIST,PMNIST or CIFAR10) and seed is the seed used for the experiment.

### 5.1.3 Data Removal & Generalization
```
python run_cifar.py 
    --dataset cifar10 
    --data_augmentation 
    --cutout 
    --sorting_file cifar10_sorted 
    --input_dir cifar10_results 
    --output_dir cifar10_results 
    --seed s 
    --remove_n r 
    --keep_lowest_n k
```
where s ranges between 1 to 5, inclusive, r is the number of training examples to remove and is in `range(0,50000,1000)`, and k is 0 (for removal by forgetting) or -1 (for random removal). This will generate a txt file and a pickle file. The txt file will contain the highest test accuracy for the run, which is used to measure generalization. 

### 5.1.4 Robustness of results across seeds
Same as Experiment 5.1.1, with different values of s. 

### 5.2.1 Dropout 
For MNIST and PMNIST, remove the `--no_dropout` flag from `run_mnist.py`. For CIFAR-10, include the `--cutout` flag in `run_cifar.py`. This requires cloning the [Cutout repo]((https://github.com/uoguelph-mlrg/Cutout)) to the same directory. 

### 5.2.2 Forgetting Sampling Frequency 
For `run_{mnist/cifar}.py`, change the value of n to a value greater than 1. 

### 5.2.3 Batch Size 
For `run_{mnist/cifar}.py`, include the flag `--batch_size b`, where b is the desired batch size. 

