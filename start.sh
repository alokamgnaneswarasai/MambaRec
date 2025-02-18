#!/bin/bash
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=20gb                     # Job memory request
#SBATCH --time=160:00:00              # Time limit hrs:min:sec

pwd; hostname; date  # Print the current working directory, hostname, and current date and time

# Run the following command
./run_exp.sh --dataset=KuaiRand2000  --train_dir=tracks256_default/ --maxlen=2048  --batch_size=16 --backbone=hourglass --hidden_units=32 --eval_neg_sample=100  --device=cuda:7 --num_epochs=500

# if you have python script, you can run it as follows
# python3 myscript.py

date