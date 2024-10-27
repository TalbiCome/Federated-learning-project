import warnings
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import prepare_dataset
from torchvision import models
from Net import Net

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader, "Training"):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, valloader):
    """Validate the model on the validation set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(valloader, "Testing"):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(valloader.dataset)
    return loss, accuracy




# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################



# Define Flower client
class FlowerClient2(fl.client.NumPyClient):

    def getModel(self, modelName):
        if modelName == "CNN":
            return Net().to(DEVICE)
        elif modelName == "ResNet18":
            return models.resnet18(pretrained=False).to(DEVICE)
        elif modelName == "ResNet34":
            return models.resnet34(pretrained=False).to(DEVICE)

    def __init__(self, cid, data_split, dataset, local_epochs, n, modelName, client_type):
        self.cid = cid
        self.net = self.getModel(modelName)
        self.epochs = local_epochs
        self.client_type = client_type
        self.trainloader, self.valloader, _ = prepare_dataset.get_data_loaderTyped(n, cid, data_split=data_split, dataset=dataset, clientType=self.client_type)
        print(f"Client {self.cid} type: {self.client_type}, trainingSetLength: {len(self.trainloader.dataset)}, model:{modelName}, data_split: {data_split}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit")
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=self.epochs)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        server_round = config["server_round"]
        print(f"[Client {self.cid}, round {server_round}] evaluate, config: {config}")
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "round": server_round}


