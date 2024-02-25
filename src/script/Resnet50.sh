
cd  ../                            # Change to working directory
module load pytorch/1.12.0              
source activate hts  

# nohup python decentralized.py   --method fedavg --aggr avg   --data cifar10 --snap 100 --rounds 300  &
nohup python decentralized.py   --method fedavg --aggr avg   --data cifar10  --snap 100 --rounds 300   --non_iid &
# nohup python decentralized.py   --method silencer --aggr avg   --data cifar10 --snap 100 --rounds 300  &
# nohup python decentralized.py   --method silencer --aggr avg   --data cifar10  --snap 100 --rounds 300   --non_iid &
wait