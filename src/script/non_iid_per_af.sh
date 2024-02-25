#!/bin/bash
#SBATCH -J lockdown                   # Job name
#SBATCH -A gts-ll72-paid               # Tracking account
#SBATCH -t 800                                    # Duration of the job (Ex: 15 mins)
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=10G
#SBATCH -q inferno                                  # Queue name (where job is submitted)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=thuang374@gatech.edu        # E-mail address for notifications

cd  ../                            # Change to working directory
module load anaconda3/2023.03
module load cuda/11.8.0
source activate hts  
# nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 0.1 --personalized --anneal_factor 0.1 & 
# nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 0.1 --personalized --anneal_factor 0.01 & 
# nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 0.1 --personalized --anneal_factor 0.001 & 
nohup python decentralized.py     --method silencer --aggr avg    --data cifar10 --non_iid  --alpha 0.1 --personalized --anneal_factor 0.00001 & 
wait