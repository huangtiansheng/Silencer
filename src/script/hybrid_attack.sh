#!/bin/bash


cd  ../                            # Change to working directory
# module load anaconda3/2023.03
# module load cuda/11.8.0

nohup python decentralized.py    --method fedavg --aggr avg    --data cifar10 --attack hybrid   &
nohup python decentralized.py    --method silencer --aggr avg    --data cifar10 --attack hybrid   &
nohup python decentralized.py    --method fedavg --aggr rlr    --data cifar10  --attack hybrid   &
nohup python decentralized.py   --method fedavg --aggr krum    --data cifar10   --attack hybrid  &
nohup python decentralized.py   --method fedavg --aggr fltrust    --data cifar10   --attack hybrid  &
wait