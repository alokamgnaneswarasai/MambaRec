#!/bin/bash

# Parse arguments from the command line
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset=*) dataset="${1#*=}";;
        --train_dir=*) train_dir="${1#*=}";;
        --maxlen=*) maxlen="${1#*=}";;
        --batch_size=*) batch_size="${1#*=}";;
        --backbone=*) backbone="${1#*=}";;
        --hidden_units=*) hidden_units="${1#*=}";;
        --eval_neg_sample=*) eval_neg_sample="${1#*=}";;
        --device=*) device="${1#*=}";;
        --num_epochs=*) num_epochs="${1#*=}";;
        *) echo "Unknown parameter: $1"; exit 1;;
    esac
    shift
done

# Validate required parameters
if [[ -z "$dataset" || -z "$backbone" || -z "$hidden_units" ]]; then
    echo "Missing required parameters. Ensure you provide --dataset, --backbone, and --hidden_units."
    exit 1
fi

# Default values for optional parameters
eval_neg_sample="${eval_neg_sample:-100}"
device="${device:-cuda:0}"

# Construct the output filename dynamically
output_dir="./termresults"
output_file="${output_dir}/${dataset}_${backbone}_hidden=${hidden_units}_batch=${batch_size}_maxlen=${maxlen}_neg=${eval_neg_sample}_device=$(echo $device | tr ':' '_')_num_epochs=${num_epochs}.out"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Run the command and redirect output to the dynamically generated file
python3 -u main.py \
  --dataset="$dataset" \
  --train_dir="$train_dir" \
  --maxlen="$maxlen" \
  --batch_size="$batch_size" \
  --backbone="$backbone" \
  --hidden_units="$hidden_units" \
  --eval_neg_sample="$eval_neg_sample" \
  --num_epochs="$num_epochs" \
  --device="$device" > "$output_file"

# Notify the user
echo "Results saved in $output_file"
