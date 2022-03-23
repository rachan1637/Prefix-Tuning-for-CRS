#!/bin/sh

export COMET_API_KEY="mjjBKjs43nOnovqf9fO5YCfYh"
export COMET_PROJECT_NAME=prefix-rec-exp
export COMET_WORKSPACE=rachan1637

source ENV/bin/activate

output_dir=outputs/lmrec_toronto_reproduce

python run.py \
  --model_name_or_path bert-base-uncased \
  --yelp_dataset_city toronto \
  --do_train \
  --do_eval \
  --do_predict \
  --max_train_samples 10 \
  --max_eval_samples 10 \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 10 \
  --learning_rate 5e-5 \
  --num_train_epochs 12 \
  --max_seq_length 400 \
  --evaluation_strategy="epoch" \
  --output_dir ${output_dir} \
  --save_total_limit 1 \
  --save_strategy="epoch" \
  --overwrite_output_dir \
  --load_best_model_at_end 