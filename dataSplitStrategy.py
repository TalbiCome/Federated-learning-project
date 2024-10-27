import torch
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np


def splitDataset(num_clients, trainset, data_split, clientType = "base"):
    if data_split == "iid":
        return iidSplit(num_clients, trainset)

    elif data_split == "non_iid_number":
        return nonIidSplit(num_clients, trainset)

    elif data_split == "non_iid_class":
        return nonIidClassSplit(num_clients, trainset)
    elif data_split == "non_iid_mobile":
        return nonIidClassSplit(num_clients, trainset)
    else:
        raise NotImplementedError("The data split is not implemented")

def iidSplit(num_clients, trainset):
    # Split training set into `num_clients` partitions to simulate different local datasets
    props = [1/num_clients]*num_clients
    datasets = random_split(trainset, props, torch.Generator().manual_seed(42))
    return datasets

def nonIidSplit(num_clients, trainset):
    props = np.random.dirichlet([0.8]*num_clients, np.random.seed(42))
    datasets = random_split(trainset, props, torch.Generator().manual_seed(42))
    print(len(datasets[0]), len(datasets[1]))
    return datasets

def nonIidClassSplit(num_clients, trainset):
    num_classes = len(trainset.classes)
    class_per_client = num_classes/num_clients
    data_indices_each_client = {client: [] for client in range(num_clients)}
    for c in range(num_classes):
        indices = (torch.tensor(trainset.targets)[..., None] == c).any(-1).nonzero(as_tuple=True)[0]
        client_belong = int(c/class_per_client)
        data_indices_each_client[client_belong].extend(list(indices))
    datasets = []
    for i in range(num_clients):
        datasets.append(Subset(trainset, data_indices_each_client[i]))
    return datasets

def nonIidMobileShrinkDatasets(num_clients, trainset, testset, clientType):
    if(clientType == "mobile"):
        trainset = keepOnlyXPercent(trainset, 0.70)
        testset = keepOnlyXPercent(testset, 0.70)
    elif(clientType == "server" or clientType == "laptop"):
        trainset = keepOnlyXPercent(trainset, 0.15)
        testset = keepOnlyXPercent(testset, 0.15)
    return testset, trainset

def nonIidServerShrinkDatasets(num_clients, trainset, testset, clientType):
    if(clientType == "mobile" or clientType == "laptop"):
        trainset = keepOnlyXPercent(trainset, 0.15)
        testset = keepOnlyXPercent(testset, 0.15)
    elif(clientType == "server" ):
        trainset = keepOnlyXPercent(trainset, 0.70)
        testset = keepOnlyXPercent(testset, 0.70)
    return testset, trainset

def keepOnlyXPercent(dataset, percent):
    dataset_size = int(percent * len(dataset))
    dataset_indices = np.random.choice(len(dataset), dataset_size, replace=False)
    return Subset(dataset, dataset_indices)