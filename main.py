import flwr as fl
from Net import Net, get_parameters
from client import FlowerClient
from server import weighted_average, fit_config
from flwr.client import Client, ClientApp
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters
from Parser import parseCmdLine

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

def client_fn(context: Context) -> Client:
    node_id = context.node_config["partition-id"]-1
    return FlowerClient(node_id, data_split, dataset, local_epochs, n, model).to_client()

def server_fn(context: Context) -> ServerAppComponents:

    # Create FedAvg strategy
    strategy = strategyFactory(strategyStr, model)
    config = ServerConfig(num_rounds=rounds)
    return ServerAppComponents(strategy=strategy, config=config)

if __name__ == "__main__":
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=n,
    )