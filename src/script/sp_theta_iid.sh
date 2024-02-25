

cd  ../                            # Change to working directory
module load anaconda3/2023.03
module load cuda/11.8.0
source activate hts  


nohup python decentralized.py  --theta 20   --method silencer --aggr avg    --data cifar10 --non_iid  &
# nohup python decentralized.py   --theta 5  --method silencer --aggr avg    --data cifar10   --non_iid &
# nohup python decentralized.py   --theta 15  --method silencer --aggr avg    --data cifar10  --non_iid &
wait