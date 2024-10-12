import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics



# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    round = metrics[0][1]["round"]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy = sum(accuracies) / sum(examples)
    print(f"Round {round} Global model test accuracy: {accuracy}")
    # Aggregate and return custom metric (weighted average)
    try:
        with open('log.txt', 'a') as f:
            if round == 1:
                f.write("\n-------------------------------------\n")
            f.write(str(accuracy)+" ")
    except FileNotFoundError:
        with open('log.txt', 'w') as f:
            if round == 1:
                f.write("\n-------------------------------------\n")
            f.write(str(accuracy)+" ")

    return {"accuracy": {accuracy}}


def fit_config(server_round:int):
    config = {
        "server_round": server_round,
    }
    return config





