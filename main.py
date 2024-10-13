import flwr as fl
from client import FlowerClient
from server import weighted_average, fit_config
from flwr.client import Client, ClientApp
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
import argparse


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
)

n = parser.parse_args().n
data_split = parser.parse_args().data_split
dataset = parser.parse_args().dataset
local_epochs = parser.parse_args().local_epochs
rounds = parser.parse_args().round
strategyStr = parser.parse_args().strategy
model = parser.parse_args().model



def client_fn(context: Context) -> Client:
    node_id = context.node_config["partition-id"]-1
    return FlowerClient(node_id, data_split, dataset, local_epochs, n, model).to_client()

def server_fn(context: Context) -> ServerAppComponents:
    def strategyFactory():
        if(strategyStr == "FedAvg"):
            strategy = fl.server.strategy.FedAvg(
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                evaluate_metrics_aggregation_fn=weighted_average
            )
        elif(strategyStr == "FedAdam"):
            strategy = fl.server.strategy.FedAvg(
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=fit_config,
                evaluate_metrics_aggregation_fn=weighted_average
            )
            
        return strategy
    # Create FedAvg strategy
    strategy = strategyFactory()
    config = ServerConfig(num_rounds=rounds)
    return ServerAppComponents(strategy=strategy, config=config)




client = ClientApp(client_fn=client_fn)
server = ServerApp(server_fn=server_fn)

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=n,
)