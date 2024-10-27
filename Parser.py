import argparse

class cmdLineInit:
    def __init__(self, n, data_split, dataset, local_epochs, rounds, strategy, model):
        self.n = n
        self.data_split = data_split
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.rounds = rounds
        self.strategy = strategy
        self.model = model

def parseCmdLine():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--round",
        type=int,
        default=10,
        help="Partition of the dataset divided into 3 iid partitions created artificially.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="The number of clients in total",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Dataset: [CIFAR10]",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="iid",
        help="iid",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="local epochs",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="FedAvg",
        help="agregation strategy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CNN",
        help="model",
        choices=["CNN", "ResNet18", "ResNet34"],
    )

    n = parser.parse_args().n
    data_split = parser.parse_args().data_split
    dataset = parser.parse_args().dataset
    local_epochs = parser.parse_args().local_epochs
    rounds = parser.parse_args().round
    strategyStr = parser.parse_args().strategy
    model = parser.parse_args().model
    return cmdLineInit(n, data_split, dataset, local_epochs, rounds, strategyStr, model)