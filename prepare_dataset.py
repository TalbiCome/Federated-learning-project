import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10
import numpy as np
from dataSplitStrategy import splitDataset

def load_datasets(num_clients: int, dataset, data_split):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset == "CIFAR10":
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    else:
        raise NotImplementedError("The dataset is not implemented")

    # Split each partition into train/val and create DataLoader
    datasets = splitDataset(num_clients, trainset, data_split)
    trainloaders, valloaders = generateTrainingValidationSet(datasets)
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader

def generateTrainingValidationSet(datasets):
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    
    return trainloaders,valloaders

def get_data_loader(num_clients: int, cid: int, dataset = "CIFAR10", data_split = "iid"):
    trainloaders, valloaders, testloader = load_datasets(num_clients, dataset, data_split)
    return trainloaders[cid], valloaders[cid], testloader