# Prefix-Tuning-for-CRS

The Repository contains the code to do keyphrase-based recommendation and explanation generation by prefix-tuning. 

For the recommendation part, the train/valid/eval can be run by the script rec.sh. For the generation part, the train/valid/eval can be run by the script run_lm.sh (Note: the GPT-2 Generation may still have some preprocessing issue, so don't use it now). The description of input arguments can be found in argument.py

The Prefix-Tuning models are all in model directory. The fundamental change of the Pretrained Language Models for Prefix-Tuning are in modeling_bart.py and modeling_gpt2.py. The files with rec are the recommmendation model, and the files with lm are the generation model. The BERT-based prefix-tuning will be updated in future.
