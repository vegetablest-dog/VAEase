import math
import os
import sys
import time
import datetime
import ml_collections
import pickle
import subprocess

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import torchvision
from torchvision import datasets as datasets
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


# Select device
def select_device():
    cuda = torch.device("cuda")
    cpu = torch.device("cpu")
    return cuda if torch.cuda.is_available() else cpu


def load_data(bs=512, device=torch.device("cuda"), ds="fmnist"):
    # Load MNIST data
    if ds == "mnist":
        mnist = datasets.MNIST(
            root="/home/ubuntu/data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        testset = datasets.MNIST(
            root="/home/ubuntu/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    else:
        mnist = datasets.FashionMNIST(
            root="/home/ubuntu/data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        testset = datasets.FashionMNIST(
            root="/home/ubuntu/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    dat = 0.001 + 0.998 * mnist.data / 255.0
    test_dat = (0.001 + 0.998 * testset.data / 255.0).to(device)
    dat = dat.to(device=device)
    loader = DataLoader(mnist, batch_size=bs, shuffle=True)

    return dat, test_dat, loader, testset.train_labels.to(device)


# Check the weights of generator
def check_weights(save_path, latent_dim=32):
    generator = torch.load(f"{save_path}/generator.pth")
    zw = torch.mean(torch.square(list(generator.upsample.parameters())[0]), dim=0)[: latent_dim]
    cw = torch.mean(torch.square(list(generator.upsample.parameters())[0]), dim=0)[latent_dim:]

    return {"z weights": zw, "c weights": cw}


# Save and load models
def save_models(path, models):
    os.makedirs(path, exist_ok=True)
    def namestr(obj, namespace=globals()):
        return [name for name in namespace if namespace[name] is obj]
    for model in models:
        torch.save(model, f"{path}/{namestr(model)[0]}.pth")

def load_models(path):
    encoder = torch.load(f"{path}/encoder.pth")
    generator = torch.load(f"{path}/generator.pth")
    loggamma = torch.load(f"{path}/loggamma.pth")
    if os.path.exists(f"{path}/prior.pth"):
        prior = torch.load(f"{path}/prior.pth")
        return encoder, generator, loggamma, prior
    return encoder, generator, loggamma

class Normalize_Tool():
    def __init__(self, max_value, min_value = 0) -> None:
        self.interval = max_value-min_value

    def forward(self, x):
        return (torch.log(x+1))/self.interval
    
    def inverse(self,x):
        return torch.exp(x*self.interval)-1

from collections import defaultdict

def variance_analysis(sigma,labels, thr = 0.05):
    N,d = sigma.shape

    sets = []
    for i in range(N):
        indices = (sigma[i] < thr).nonzero(as_tuple=True)[0]  
        sets.append(set(indices.tolist())) 


    def set_distance(set1, set2):
        return len(set1 | set2) - len(set1 & set2) 


    grouped_sets = defaultdict(list)
    for i in range(N):
        grouped_sets[labels[i].item()].append(sets[i])


    intra_group_distances = []
    inter_group_distances = []


    for group in grouped_sets.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                distance = set_distance(group[i], group[j])
                intra_group_distances.append(distance)


    labels_list = list(grouped_sets.keys())
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            group1 = grouped_sets[labels_list[i]]
            group2 = grouped_sets[labels_list[j]]
            for set1 in group1:
                for set2 in group2:
                    distance = set_distance(set1, set2)
                    inter_group_distances.append(distance)


    intra_group_avg = np.mean(intra_group_distances) if intra_group_distances else 0
    inter_group_avg = np.mean(inter_group_distances) if inter_group_distances else 0

    return intra_group_avg,inter_group_avg

def variance_matrix_analysis(var,labels):
    N,d = var.shape

    sets = []
    for i in range(N):
        indices = (var[i] < 0.05).nonzero(as_tuple=True)[0] 
        sets.append(set(indices.tolist()))  


    def set_distance(set1, set2):
        return len(set1 | set2) - len(set1 & set2)  


    grouped_sets = defaultdict(list)
    for i in range(N):
        grouped_sets[labels[i].item()].append(sets[i])


    intra_group_distances = defaultdict(list)  
    inter_group_distances = defaultdict(lambda: defaultdict(list)) 


    for group_label, group in grouped_sets.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                distance = set_distance(group[i], group[j])
                intra_group_distances[group_label].append(distance)


    labels_list = list(grouped_sets.keys())
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            group1 = grouped_sets[labels_list[i]]
            group2 = grouped_sets[labels_list[j]]
            for set1 in group1:
                for set2 in group2:
                    distance = set_distance(set1, set2)
                    inter_group_distances[labels_list[i]][labels_list[j]].append(distance)
                    inter_group_distances[labels_list[j]][labels_list[i]].append(distance)  


    matrix = np.zeros((10, 10))  

    for group_label, distances in intra_group_distances.items():
        matrix[group_label, group_label] = np.mean(distances)


    for i in range(10):
        for j in range(i + 1, 10):
            if (i != j):  
                if (i in inter_group_distances) and (j in inter_group_distances[i]):
                    matrix[i, j] = np.mean(inter_group_distances[i][j])
                    matrix[j, i] = matrix[i, j] 

    return matrix


def find_optimal_threshold_torch(array):
    if array.numel() < 2:
        return None
    

    sorted_array = torch.sort(array).values
    n = sorted_array.size(0)
    

    cumulative_sum = torch.cumsum(sorted_array, dim=0)
    cumulative_sq_sum = torch.cumsum(sorted_array**2, dim=0)
    
    min_variance_sum = float('inf')
    optimal_threshold = sorted_array[0]


    for i in range(1, n):

        count_lower = i
        sum_lower = cumulative_sum[i - 1]
        sq_sum_lower = cumulative_sq_sum[i - 1]
        var_lower = (sq_sum_lower - sum_lower**2 / count_lower) / count_lower
        
        count_upper = n - i
        sum_upper = cumulative_sum[-1] - sum_lower
        sq_sum_upper = cumulative_sq_sum[-1] - sq_sum_lower
        var_upper = (sq_sum_upper - sum_upper**2 / count_upper) / count_upper
        
        variance_sum = var_lower * count_lower + var_upper * count_upper
        
        if variance_sum < min_variance_sum:
            min_variance_sum = variance_sum
            optimal_threshold = sorted_array[i - 1]
    
    return optimal_threshold.item() 