

cd  ../                            # Change to working directory
module load anaconda3/2023.03
module load cuda/11.8.0
source activate hts  
nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --pruning  random --dis_check_gradient  &
nohup python decentralized.py     --method silencer --aggr avg    --data cifar10  --non_iid --pruning  random --dis_check_gradient &
wait