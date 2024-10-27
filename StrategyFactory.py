import flwr as fl
from server import weighted_average, fit_config
from Net import Net, get_parameters
from torchvision import models
from flwr.common import ndarrays_to_parameters
from typing import List, Tuple
from flwr.common import Metrics

def strategyFactory(strategyStr, modelString):
    if(strategyStr == "FedAvg"):
        return instanciateFedAvg()
    elif(strategyStr == "FedAvgWithModel"):
        return instanciateFedAvgWithModel(modelString)
    elif(strategyStr == "FedAdam"):
        return instanciateFedAdam(modelString)
    else:
        raise NotImplementedError("The strategy " + strategyStr + " is not implemented")

def instanciateFedAvg():
    return fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average
    )

class specialFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, modelString, numRound):
        self.params = getModelParameters(modelString)
        self.result = None
        self.numRound = numRound
        super().__init__(
            initial_parameters=ndarrays_to_parameters(self.params),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=self.weighted_average
        )

    def weighted_average(self, metrics: List[Tuple[int, Metrics]]):
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
        if(round == self.numRound):
            self.result = accuracy
        return {"accuracy": {accuracy}}

def instanciateFedAdam(modelString):
    params = getModelParameters(modelString)
    return fl.server.strategy.FedAdam(
        initial_parameters=ndarrays_to_parameters(params),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        eta=0.01,
        eta_l=0.0316,
        beta_1=0.9,
        beta_2=0.99,
        tau=0.001,
    )

def getModelParameters(modelString):
    if(modelString == "CNN"):
        return get_parameters(Net())
    elif(modelString == "ResNet18"):
        return get_parameters(models.resnet18(pretrained=False))
    elif(modelString == "ResNet34"):
        return get_parameters(models.resnet34(pretrained=False))
    else:
        raise NotImplementedError("The model " + modelString + " is not implemented")