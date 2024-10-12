# Project in Federated learning
The project should be handled including:
- pdf for experimental results (plots) and explanations (covering the bullet points of Section 1 and 2)
- source code 

on Moodle by October 28th. 

## Installation via conda 
[This step can be skipped if you have installed the packages]:
* Initialize the conda environment for the course
* [PyTorch installation](https://pytorch.org/)
* Install the Flower package
  ```bash
  > conda config --add channels conda-forge
  > conda config --set channel_priority strict
  > conda install flwr[simulation]
  > pip install tqdm
  ```

## Testez:
* Open one terminal in PyCharm 
```bash
> python main.py --n 10
``` 

## Warm up exercises:
* Complete TP1: If you have not yet finished TP1, please complete it before continuing with this project.
* In this project, we are simulating Federated Learning workloads using the run_simulation() function provided by Flower. Review the code in main.py to understand its functionality. For further insight, refer to this [this tutorial](https://flower.ai/docs/framework/tutorial-series-use-a-federated-learning-strategy-pytorch.html).
* If, in each round, the server samples only half of the clients and only these selected clients return their evaluations, how can you implement this behavior in the code?
   **Hint**: Check the code explanation for [FedAvg](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py) Strategy.

   By changing the parameter fraction_fit when instanciating the fedAvg strategy

## 1. FedAvg vs FedAdam
[FedAdam](https://arxiv.org/pdf/2003.00295) is a federated learning algorithm proposed by Google in 2021 that has been shown to converge faster than FedAvg.

* First, study the algorithm of FedAdam and note the key differences between FedAvg and FedAdam.
  
  **Question**: Is the FedAdam algorithm equivalent to an algorithm where each client uses the Adam optimizer for local computation? Under what circumstances, if any, would these two algorithms behave the same?

  No because the fedAdam agregates the model of each clients so it's not similar to a local agregation.
  If all clients have the same data

* Add a new argument, "strategy," to the code, allowing it to execute either the FedAvg strategy or the FedAdam [FedAdam strategy](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAdam.html) algorithm.
For example:
```bash
> python main.py --n 10 --strategy "FedAdam" #The code will run experiments using FedAdam algorithm. 
``` 



* Compare the performance of FedAvg and FedAdam. For hyperparameter tuning, refer to the [Section D.1](https://arxiv.org/pdf/2003.00295).
* Produce a plot similar to [Figure 1](https://arxiv.org/pdf/2003.00295) and provide an explanation of the results.
* Experiment in the "non_iid_class" setting and provide your observations.
* Compare your results with the conclusions of the paper. If your results differ, explain the reasons for the discrepancies.


## 2. System Heterogeneity
Consider a scenario where we have 9 clients in a federated setting. These clients are divided into three categories based on device capacity: three are mobile devices, three are laptops, and three are servers. They can be clustered based on their memory and computational capacities.


Mobiles can only handle a CNN model. Laptops can hold a [Resnet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
and server can support a [Resnet34](https://pytorch.org/vision/main/models/resnet.html). 

* Modify the code to add a "model" argument that specifies the model architecture for training. Options should include "CNN," "ResNet18," and "ResNet34."

Learning Methods:

1. FedAvg: All clients train the CNN model using the FedAvg strategy. The final global model is used for inference.
2. Cluster-FedAvg: Clients within the same cluster train the largest model their devices can accommodate using FedAvg. Each cluster uses its final model for inference.
3. Cluster-FedAvg-Ensemble: Clients within the same cluster train the largest model their devices can support using FedAvg. For inference, they use an ensemble approach (e.g., majority voting).


* Implement a new data split strategy called "non_iid_mobiles", where mobile clients have access to 70% of the sample points, while servers and laptops each have access to 15%.
* Implement a "non_iid_server" data split, where servers have access to 70% of the sample points, while mobile clients and laptops each have access to 15%.
* Add a metric to log the final accuracy for each cluster.
* Run FedAvg under the following settings: "iid," "non_iid_mobiles," and "non_iid_server." Compare the model performance across each cluster and provide an explanation for the results.
* Implement Cluster-FedAvg:

    Hint: You can simulate three separate instances of FedAvg with 3 clients each. Be cautious with the dataset split! Alternatively, you may need to re-implement the FedAvg strategy (refer to the FedAdam code).

* Compare the performance of Cluster-FedAvg with FedAvg under the "iid" and "non_iid_mobiles" settings. Provide your analysis.
   
    **Hints**: we can simulate three times the separate FedAvg with 3 clients, but be careful on the dataset split! Or we need to reimplement the FedAvg strategy, refer to FedAdam code. 
* Compare Cluster-FedAvg with FedAvg on the setting "iid, non_iid_mobiles", provide your analysis
* *Bonus*: Implement Cluster-FedAvg-Ensemble and compare its performance with both Cluster-FedAvg and FedAvg across various data split settings.
           ```