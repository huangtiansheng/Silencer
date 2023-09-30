

# Silencer 
This is the repo for the paper "Silencer: Pruning-aware Backdoor Defense for Decentralized Federated Learning".

## Algorithm overview
The overall procedure can be summarized into four main steps. i)Pruning-aware training. ii) Mask searching. iii) Aggregation on each client. iv) After T rounds, do consensus filtering.



## Package requirement
* PyTorch 
* Numpy
* TorchVision

## Data  preparation
Dataset FashionMnist and CIFAR10/100 will be automatically downloaded with TorchVision.

## Command to run
The following code run lockdown in its default setting:
```
python decentralized.py  
```

## Files organization
The main simulation is in `decentralized.py`, where we initialize the benign and poison dataset, call clients to do local training, call aggregator to do aggregation, do consensus filtering before testing, etc.

The client local training logistic is in `agent_bi.py`. It basically involves pruning-aware training and mask searching procedure.  

The aggregation logistic is in `aggregation.py`, where we implement multiple defense baselines. 

The data poisoning, data preparation and data distribution logistic is in `utils.py`. Specifically, check the function `poison_dataset()` of how we inject backdoor to the data. 


## Logging and checkpoint
The logging files will be contained in `src/logs`. Benign accuracy, ASR, and Backdoor accuracy will be tested in every 50 rounds.
For Lockdown, the three metrics correspond to the following logging format:
```
| Clean Val_Loss/Val_Acc:  (Benign accuracy) |
| Clean Attack Success Ratio:  (ASR) |
| Clean Poison Loss/Clean Poison accuracy: (Backdoor Acc)|
```
Model checkpoints will be saved every 50 rounds in the directory `src/checkpoint`.







