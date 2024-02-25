#!/bin/bash
#SBATCH -J lockdown                   # Job name
#SBATCH -A gts-ll72-paid               # Tracking account
#SBATCH -t 1000                                    # Duration of the job (Ex: 15 mins)
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-cpu=10G
#SBATCH -q inferno                                  # Queue name (where job is submitted)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=thuang374@gatech.edu        # E-mail address for notifications
cd  ../                            # Change to working directory
module load pytorch/1.12.0              
source activate hts  

nohup python decentralized.py   --method fedavg --aggr avg   --data fmnist  --snap 100  --topology full --non_iid  &
nohup python decentralized.py   --method fedavg --aggr avg   --data fmnist  --snap 100  --attack DBA --non_iid    --topology full  &
# nohup python decentralized.py   --method fedavg --aggr avg   --data cifar10   --snap 100 --rounds 300  --source_class 7 --non_iid --poison_frac 1  &

# nohup python decentralized.py   --method silencer --aggr avg   --data cifar10  --snap 100 --rounds 300  --source_class 7 --non_iid --poison_frac 1  &
wait