#!/bin/bash


cd  ../                            # Change to working directory
module load anaconda3/2023.03
module load cuda/11.8.0
nohup python decentralized.py    --method silencer --aggr avg    --data cifar10   --non_iid --alpha 0.1  --personalized &
nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 0.3 --personalized & 
nohup python decentralized.py    --method silencer --aggr avg    --data cifar10  --non_iid --alpha 0.5 --personalized &
nohup python decentralized.py    --method silencer --aggr avg    --data cifar10   --non_iid --alpha 0.7 --personalized &
nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 1 --personalized &
wait