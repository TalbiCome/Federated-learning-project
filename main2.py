import flwr as fl
from Net import Net, get_parameters
from client2 import FlowerClient2
from server import weighted_average, fit_config
from flwr.client import Client, ClientApp
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters
from Parser import parseCmdLine
from StrategyFactory import specialFedAvg

from torchvision import models
from StrategyFactory import strategyFactory

parser = parseCmdLine()
n = parser.n
data_split = parser.data_split
dataset = parser.dataset
local_epochs = parser.local_epochs
rounds = parser.rounds
strategyStr = parser.strategy
model = parser.model

n = 3

def client_fn(context: Context) -> Client:
    node_id = context.node_config["partition-id"]-1
    return FlowerClient2(node_id, data_split, dataset, local_epochs, n, model).to_client()

def client_fn1(context: Context) -> Client:
    node_id = context.node_config["partition-id"]-1
    return FlowerClient2(node_id, data_split, dataset, local_epochs, n, "CNN", "mobile").to_client()

def client_fn2(context: Context) -> Client:
    node_id = context.node_config["partition-id"]-1
    return FlowerClient2(node_id, data_split, dataset, local_epochs, n, "ResNet18", "laptop").to_client()

def client_fn3(context: Context) -> Client:
    node_id = context.node_config["partition-id"]-1
    return FlowerClient2(node_id, data_split, dataset, local_epochs, n, "ResNet34", "server").to_client()

strategy1 = specialFedAvg("CNN", rounds)
def server_fn1(context: Context) -> ServerAppComponents:
    # Create FedAvg strategy
    strategy = strategy1
    config = ServerConfig(num_rounds=rounds)
    return ServerAppComponents(strategy=strategy, config=config)

strategy2 = specialFedAvg("ResNet18", rounds)
def server_fn2(context: Context) -> ServerAppComponents:
    # Create FedAvg strategy
    
    config = ServerConfig(num_rounds=rounds)
    return ServerAppComponents(strategy=strategy2, config=config)

strategy3 = specialFedAvg("ResNet34", rounds)
def server_fn3(context: Context) -> ServerAppComponents:
    # Create FedAvg strategy
    
    config = ServerConfig(num_rounds=rounds)
    return ServerAppComponents(strategy=strategy3, config=config)

if __name__ == "__main__":
    # Run simulation
    # run_simulation(
    #     server_app=ServerApp(server_fn=server_fn1),
    #     client_app=ClientApp(client_fn=client_fn1),
    #     num_supernodes=n,
    # )
    
    # run_simulation(
    #     server_app=ServerApp(server_fn=server_fn2),
    #     client_app=ClientApp(client_fn=client_fn2),
    #     num_supernodes=n,
    # )

    run_simulation(
        server_app=ServerApp(server_fn=server_fn3),
        client_app=ClientApp(client_fn=client_fn3),
        num_supernodes=n,
    )

    print("Simulation done!")
    print("results = " + "mobile: " + str(strategy1.result) + "laptop: " + str(strategy2.result) + "server: " + str(strategy3.result))