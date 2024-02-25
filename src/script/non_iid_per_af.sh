#!/bin/bash


cd  ../                            # Change to working directory
# module load anaconda3/2023.03
# module load cuda/11.8.0
# nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 0.1 --personalized --anneal_factor 0.1 & 
# nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 0.1 --personalized --anneal_factor 0.01 & 
# nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 0.1 --personalized --anneal_factor 0.001 & 
nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 0.1 --personalized --anneal_factor 0.00001 & 
wait