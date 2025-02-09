

cd  ../                            # Change to working directory
module load anaconda3/2023.03
module load cuda/11.8.0
source activate hts  

nohup python decentralized.py   --method fedavg --aggr avg    --data cifar10  --non_iid &
nohup python decentralized.py   --theta 25  --method silencer --aggr avg    --data cifar10   --non_iid &
nohup python decentralized.py   --theta 30   --method silencer --aggr avg    --data cifar10  --non_iid  &
nohup python decentralized.py   --theta 35   --method silencer --aggr avg    --data cifar10  --non_iid  &
wait