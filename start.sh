#!/bin/bash
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=160:00:00              # Time limit hrs:min:sec

pwd; hostname; date  # Print the current working directory, hostname, and current date and time

# Run the following command
./run_exp.sh --dataset=ml-1m  --train_dir=tracks256_default/ --maxlen=200  --batch_size=16 --backbone=samba --hidden_units=64 --eval_neg_sample=100  --device=cuda:4 --num_epochs=500

date