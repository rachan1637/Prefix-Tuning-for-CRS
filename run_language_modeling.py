#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Explanation Generation by prefix-tuning
"""
import logging
import sys
import os
import pickle

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ConstantLR

import transformers
from transformers import (
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    BartTokenizerFast,
    GPT2TokenizerFast,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
# import datasets

from arguments import ModelArguments, DataTrainingArguments
from data_utils import load_table2text_dataset, DataCollatorForTable2Text
from model.modeling_bart import BartForConditionalGeneration
from model.modeling_gpt2 import GPT2LMHeadModel
from model.bart_lm_prefix_model import PrefixTuning_BartforLM
from model.gpt2_lm_prefix_model import PrefixTuning_GPT2ForLM
from trainer import EvalPrediction, EvaluateFriendlySeq2SeqTrainer
from metrics import compute_bleu, compute_bleu_metric, compute_rouge_metric

logger = logging.getLogger(__name__)

def main():
    os.environ["WANDB_DISABLED"] = 'true'
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # All these arguments can be found in arugments.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(model_args)
    logger.info(data_args)

    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_file is not None:
        with open(data_args.dataset_file, 'rb') as fin:
            dataset = pickle.load(fin)

        if training_args.do_train:
            train_dataset = load_table2text_dataset(
                dataset.input_ids_train, dataset.attention_mask_train, dataset.labels_train, dataset.item_labels_train, dataset.user_labels_train, data_args.max_train_samples
            )
        
        if training_args.do_eval:
            eval_dataset = load_table2text_dataset(
                dataset.input_ids_eval, dataset.attention_mask_eval, dataset.labels_eval, dataset.item_labels_eval, dataset.user_labels_eval, data_args.max_eval_samples
            )

        if training_args.do_predict:
            test_dataset = load_table2text_dataset(
                dataset.input_ids_test, dataset.attention_mask_test, dataset.labels_test, dataset.item_labels_test, dataset.user_labels_test, data_args.max_predict_samples
            )
    else:
        raise ValueError("Please specify the dataset file")
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.model_type == 'gpt2':
        config.pad_token_id = 50257
    elif model_args.model_type == "bart":
        config.pad_token_id = 1
    
    if model_args.tuning_mode == "prefixtune":
        if model_args.prefix_seq_len == 0:
            raise ValueError("Please specify prefix_seq_len")
        if model_args.mid_dim == 0:
            raise ValueError("Please specify mid_dim (512 in general)")
        if model_args.with_interaction is None:
            raise ValueError("Please specify whether the interaction layer should be added")

        config.preseqlen = model_args.prefix_seq_len
        config.mid_dim = model_args.mid_dim
        config.with_interaction = model_args.with_interaction
        config.tuning_mode = "prefixtune" # useless? probably drop in future
        config.add_item_prefix = model_args.add_item_prefix
    elif model_args.tuning_mode == "finetune":
        config.tuning_mode = "finetune" # same as above
    else:
        raise ValueError(f"Unrecoginized tuning_mode {model_args.tuning_mode}")

    if model_args.num_users != 0:
        config.num_users = model_args.num_users
    else:
        config.num_users = len(dataset.user_id_list)
        logger.warning("You are giving num_users based on the dataset, it's better to check and specify by yourself.")
    
    if model_args.num_items != 0:
        config.num_items = model_args.num_items
    else:
        config.num_items = len(dataset.item_id_list)
        logger.warning("You are giving num_items based on the dataset, it's better to check and specify by yourself.")
    
    if model_args.model_type == "gpt2":
        if model_args.tuning_mode == "finetune":
            model = GPT2LMHeadModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            # We add one additional [PAD] token during the tokenization. Need to resize the embedding.
            model.resize_token_embeddings(50258)
            logger.info("Initialize Fine Tuning GPT2 for LM successfully")
        elif model_args.tuning_mode == "prefixtune":
            if model_args.model_name_or_path == "gpt2":
                pretrained_model = GPT2LMHeadModel.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
                # We add one additional [PAD] token during the tokenization. Need to resize the embedding.
                pretrained_model.resize_token_embeddings(50258)

                # Freeze Bart Parameter
                logger.info("Freeze GPT2 parameters")
                for n, param in pretrained_model.named_parameters():
                        # Only freeze the lm part, not the head.
                        if "transformer" in n:
                            param.requires_grad = False
                model = PrefixTuning_GPT2ForLM(config, gpt2_model=pretrained_model)
                logger.info("Initialize Prefix Tuning GPT2 for LM successfully")
            else:
                model = GPT2LMHeadModel.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )          
                logger.info("Load Prefix Tuning GPT2 for LM successfully")
    elif model_args.model_type == "bart":
        if model_args.tuning_mode == "finetune":
            model = BartForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            logger.info("Initialize Fine Tuning Bart for LM successfully")
        elif model_args.tuning_mode == "prefixtune":
            if model_args.model_name_or_path == "facebook/bart-base":
                pretrained_model = BartForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
                # Freeze Bart Parameter
                logger.info("Freeze Bart parameters")
                for n, param in pretrained_model.named_parameters():
                        # Only freeze the lm part, not the head.
                        if "model" in n:
                            param.requires_grad = False
                model = PrefixTuning_BartforLM(config, bart_model=pretrained_model)
                logger.info("Initialize Prefix Tuning Bart for LM successfully")
            else:
                model = PrefixTuning_BartforLM.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
                logger.info("Load Prefix Tuning Bart for LM successfully")
    else:
        raise ValueError("We only allow gpt2 and bart as our base model.")
    
    # lm_blue_metric = datasets.load_metric("bleu")
    # lm_rouge_metric = datasets.load_metric("rouge")
    def compute_metrics(eval_predictions: EvalPrediction, section: str):
        # Compute BLEU score
        # bleu_4 = lm_blue_metric.compute(predictions = eval_predictions.predictions, references=eval_predictions.items)
        # bleu_1 = lm_blue_metric.compute(predictions = eval_predictions.predictions, references=eval_predictions.items, max_order=1)
        bleu_4 = compute_bleu_metric(predictions = eval_predictions.predictions, references=eval_predictions.items)
        bleu_1 = compute_bleu_metric(predictions = eval_predictions.predictions, references=eval_predictions.items, max_order=1)
        output = {"bleu_1": bleu_1["bleu"], "bleu_4": bleu_4["bleu"]}

        # Compute Rouge score
        predictions_str = [' '.join(pred).strip() for pred in eval_predictions.predictions]
        references_str = [' '.join(ref[0]).strip() for ref in eval_predictions.items]
        # rouge = lm_rouge_metric.compute(predictions=predictions_str, references=references_str)
        rouge = compute_rouge_metric(predictions=predictions_str, references=references_str)
        rouge_1 = {
            "r1_p": rouge["rouge1"].mid.precision, 
            "r1_r": rouge["rouge1"].mid.recall, 
            "r1_f": rouge["rouge1"].mid.fmeasure
        }
        rouge_2 = {
            "r2_p": rouge["rouge2"].mid.precision, 
            "r2_r": rouge["rouge2"].mid.recall, 
            "r2_f": rouge["rouge2"].mid.fmeasure
        }
        output = {**output, **rouge_1, **rouge_2}
        return output

    data_collator = DataCollatorForTable2Text(
        tuning_mode = model_args.tuning_mode,
        add_item_prefix = model_args.add_item_prefix,
        prefix_only = model_args.prefix_only
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if model_args.tuning_mode == "prefixtune": 
        name_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"All params that will be optimized are {name_params}")
    optimizer = AdamW(params=params, lr=training_args.learning_rate)
    # scheduler = StepLR(optimizer, step_size=0, gamma=0.0)
    scheduler = ConstantLR(optimizer)

    if "gpt2" == model_args.model_type:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif 'bart' == model_args.model_type:
        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

    trainer = EvaluateFriendlySeq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=dataset.review_text_eval if data_args.max_eval_samples is None else dataset.review_text_eval[:data_args.max_eval_samples],
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset=test_dataset if test_dataset else eval_dataset,
            test_examples=dataset.review_text_test if data_args.max_predict_samples is None else dataset.review_text_test[:data_args.max_predict_samples],
            metric_key_prefix="predict"
        )
        metrics = predict_results.metrics
        max_predict_samples = len(test_dataset)
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()