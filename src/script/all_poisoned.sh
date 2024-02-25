#!/bin/bash
#SBATCH -J lockdown                   # Job name
#SBATCH -A gts-ll72-paid               # Tracking account
#SBATCH -t 700                                    # Duration of the job (Ex: 15 mins)
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-cpu=10G
#SBATCH -q inferno                                  # Queue name (where job is submitted)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=thuang374@gatech.edu        # E-mail address for notifications
cd  ../                            # Change to working directory
module load pytorch/1.12.0               
source activate hts  
nohup python decentralized.py   --method silencer --aggr avg   --data cifar10  --num_corrupt 40 --poison_frac 0.01 --snap 150 --anneal_factor 0.001 &
nohup python decentralized.py   --method silencer --aggr avg   --data cifar10  --num_corrupt 40 --poison_frac 0.01  --snap 150  --anneal_factor 0.001  --non_iid &
nohup python decentralized.py   --method silencer --aggr avg   --data cifar10  --num_corrupt 40 --poison_frac 0.01 --snap 150 --anneal_factor 0.1 &
nohup python decentralized.py   --method silencer --aggr avg   --data cifar10  --num_corrupt 40 --poison_frac 0.01  --snap 150  --anneal_factor 0.1  --non_iid &
wait