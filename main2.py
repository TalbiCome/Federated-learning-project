import flwr as fl
from Net import Net, get_parameters
from client2 import FlowerClient2
from server import weighted_average, fit_config
from flwr.client import Client, ClientApp
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters
from Parser import parseCmdLine
from StrategyFactory import clusterFedAvg

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

class clusteredFedAvg():
    def __init__(self, rounds):
        self.strategy1 = clusterFedAvg("CNN", rounds)
        self.strategy2 = clusterFedAvg("ResNet18", rounds)
        self.strategy3 = clusterFedAvg("ResNet34", rounds)
        self.strategy4 = clusterFedAvg("CNN", rounds)
        self.strategy5 = clusterFedAvg("CNN", rounds)

    def server_fn1(self, context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=rounds)
        return ServerAppComponents(strategy=self.strategy1, config=config)

    def server_fn2(self, context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=rounds)
        return ServerAppComponents(strategy=self.strategy2, config=config)

    def server_fn3(self, context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=rounds)
        return ServerAppComponents(strategy=self.strategy3, config=config)
    
    def server_fn4(self, context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=rounds)
        return ServerAppComponents(strategy=self.strategy4, config=config)

    def server_fn5(self, context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=rounds)
        return ServerAppComponents(strategy=self.strategy5, config=config)
    
    def client_fn1(self, context: Context) -> Client:
        node_id = context.node_config["partition-id"]-1
        return FlowerClient2(node_id, data_split, dataset, local_epochs, n, "CNN", "mobile").to_client()

    def client_fn2(self, context: Context) -> Client:
        node_id = context.node_config["partition-id"]-1
        return FlowerClient2(node_id, data_split, dataset, local_epochs, n, "ResNet18", "laptop").to_client()

    def client_fn3(self, context: Context) -> Client:
        node_id = context.node_config["partition-id"]-1
        return FlowerClient2(node_id, data_split, dataset, local_epochs, n, "ResNet34", "server").to_client()
    
    def client_fn4(self, context: Context) -> Client:
        node_id = context.node_config["partition-id"]-1
        return FlowerClient2(node_id, data_split, dataset, local_epochs, n, "CNN", "laptop").to_client()
    
    def client_fn5(self, context: Context) -> Client:
        node_id = context.node_config["partition-id"]-1
        return FlowerClient2(node_id, data_split, dataset, local_epochs, n, "CNN", "server").to_client()

    def run_MultiModelSimulation(self):
        run_simulation(
            server_app=ServerApp(server_fn=self.server_fn1),
            client_app=ClientApp(client_fn=self.client_fn1),
            num_supernodes=n,
        )
        
        run_simulation(
            server_app=ServerApp(server_fn=self.server_fn2),
            client_app=ClientApp(client_fn=self.client_fn2),
            num_supernodes=n,
        )

        run_simulation(
            server_app=ServerApp(server_fn=self.server_fn3),
            client_app=ClientApp(client_fn=self.client_fn3),
            num_supernodes=n,
        )

    def runCNNOnlySimulation(self):
        run_simulation(
            server_app=ServerApp(server_fn=self.server_fn1),
            client_app=ClientApp(client_fn=self.client_fn1),
            num_supernodes=n,
        )

        run_simulation(
            server_app=ServerApp(server_fn=self.server_fn4),
            client_app=ClientApp(client_fn=self.client_fn4),
            num_supernodes=n,
        )

        run_simulation(
            server_app=ServerApp(server_fn=self.server_fn5),
            client_app=ClientApp(client_fn=self.client_fn5),
            num_supernodes=n,
        )

    def printMultiModelSimulationOutput(self):
        print("Simulation done!")
        print("results = " + "mobile: " + str(self.strategy1.result) + "laptop: " + str(self.strategy2.result) + "server: " + str(self.strategy3.result))

    def printCNNOnlySimulationOutput(self):
        print("Simulation done!")
        print("results = " + "mobile: " + str(self.strategy1.result) + "laptop: " + str(self.strategy4.result) + "server: " + str(self.strategy5.result))

if __name__ == "__main__":

    useClusteredMultimodelFedAvg = False

    if(useClusteredMultimodelFedAvg):
        obj = clusteredFedAvg(rounds)
        obj.run_MultiModelSimulation()
        obj.printMultiModelSimulationOutput()
    else:
        obj = clusteredFedAvg(rounds)
        obj.runCNNOnlySimulation()
        obj.printMultiModelSimulationOutput()