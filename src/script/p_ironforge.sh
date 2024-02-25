

cd  ../                            # Change to working directory
module load anaconda3/2023.03
module load cuda/11.8.0
source activate hts  

nohup python decentralized.py    --method fedavg    --aggr ironforge   --poison_frac 0.8  --data cifar10  &
nohup python decentralized.py    --method fedavg  --aggr ironforge   --poison_frac 0.5  --data cifar10   &
nohup python decentralized.py   --method fedavg   --aggr ironforge  --poison_frac 0.2  --data cifar10  &
nohup python decentralized.py   --method fedavg   --aggr ironforge  --poison_frac 0.05  --data cifar10  &
nohup python decentralized.py   --method fedavg    --aggr ironforge  --poison_frac 0  --data cifar10   &
wait