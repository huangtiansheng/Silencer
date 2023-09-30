

# Lockdown: Backdoor Defense for Federated Learning with Isolated Subspace Training
This is the repo for the paper "Lockdown: Backdoor Defense for Federated Learning with Isolated Subspace Training".

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

## Logging and checkpoint
The logging files will be contained in `src/logs`. Benign accuracy, ASR, and Backdoor accuracy will be tested in every 50 rounds.
For Lockdown, the three metrics correspond to the following logging format:
```
| Clean Val_Loss/Val_Acc:  (Benign accuracy) |
| Clean Attack Success Ratio:  (ASR) |
| Clean Poison Loss/Clean Poison accuracy: (Backdoor Acc)|
```
Model checkpoints will be saved every 50 rounds in the directory `src/checkpoint`.







