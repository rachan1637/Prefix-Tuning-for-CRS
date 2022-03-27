#!/bin/sh

# export COMET_API_KEY="mjjBKjs43nOnovqf9fO5YCfYh"
# export COMET_PROJECT_NAME=prefix-rec-exp
# export COMET_WORKSPACE=rachan1637

source ENV/bin/activate

# model_name_or_path=/home/chanyunh/projects/def-ssanner/chanyunh/lmrec_re/outputs/lmrec_reprod_check
output_dir=/home/chanyunh/projects/def-ssanner/chanyunh/lmrec_re/outputs/trt/bart_keyphrase_prefixtune_len_5
# output_dir=outputs/trash

CUDA_LAUNCH_BLOCKING=1 python run.py \
  --model_name_or_path facebook/bart-base \
  --model_type bart \
  --yelp_dataset_city toronto \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 22 \
  --per_device_eval_batch_size 22 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --max_seq_length 400 \
  --evaluation_strategy="epoch" \
  --output_dir ${output_dir} \
  --save_total_limit 1 \
  --save_strategy="epoch" \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --num_labels 1121 \
  --input_data_mode keyphrase \
  --tuning_mode prefixtune \
  --prefix_seq_len 5 \
  --mid_dim 512 \
  --num_users 1073
  # --num_users 1073 \