import flwr as fl
from server import weighted_average, fit_config
from Net import Net, get_parameters
from torchvision import models
from flwr.common import ndarrays_to_parameters

def strategyFactory(strategyStr, modelString):
    if(strategyStr == "FedAvg"):
        return instanciateFedAvg()
    
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