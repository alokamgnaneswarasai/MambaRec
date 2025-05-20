#!/bin/bash
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=20gb                     # Job memory request
#SBATCH --time=160:00:00              # Time limit hrs:min:sec

pwd; hostname; date  # Print the current working directory, hostname, and current date and time

# Run the following command
# ./run_exp.sh --dataset=KuaiRand2000 --train_dir=tracks256_default/ --maxlen=1536 --batch_size=32 --backbone=localmamba --hidden_units=32 --eval_neg_sample=100  --device=cuda:3 --num_epochs=500
# ./run_exp.sh --dataset=tracks2000 --train_dir=tracks256_default/ --maxlen=1536 --batch_size=32 --backbone=qlocalmamba --hidden_units=32 --eval_neg_sample=100  --device=cuda:0 --num_epochs=500
./run_exp.sh --dataset=ratings_Beauty --train_dir=tracks256_default/ --maxlen=5 --batch_size=32 --backbone=sas --hidden_units=32 --eval_neg_sample=100  --device=cuda:6 --num_epochs=500

# if you have python script, you can run it as follows
# python3 myscript.py

date