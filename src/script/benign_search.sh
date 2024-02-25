#!/bin/bash
#SBATCH -J lockdown                   # Job name
#SBATCH -A gts-ll72               # Tracking account
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-cpu=10G
#SBATCH -q embers                                  # Queue name (where job is submitted)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=thuang374@gatech.edu        # E-mail address for notifications
cd  ../                            # Change to working directory
module load pytorch/1.12.0                  # Load module dependencies
source activate hts  
nohup python decentralized.py   --method silencer --aggr avg   --data cifar10 --attack benign_search --snap 150 --anneal_factor 0.1 &
nohup python decentralized.py   --method silencer --aggr avg   --data cifar10 --attack benign_search  --snap 150   --anneal_factor 0.1 --non_iid &
nohup python decentralized.py   --method silencer --aggr avg   --data cifar10 --attack benign_search --snap 150 --anneal_factor 0.01  &
nohup python decentralized.py   --method silencer --aggr avg   --data cifar10 --attack benign_search  --snap 150   --anneal_factor 0.01 --non_iid &
wait