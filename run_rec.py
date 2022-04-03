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
Recommendation by prefix-tuning
"""
import logging
import os
import sys
# import comet_ml
import pickle
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ConstantLR
from sklearn.metrics import (
    f1_score, 
    recall_score, 
    precision_score,
    top_k_accuracy_score,
)
from scipy.special import softmax

import transformers
from transformers import (
    AutoConfig,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction

from data_utils import (
    DataCollatorForYelpRec, 
    load_rec_dataset
)

from arguments import ModelArguments, DataTrainingArguments
from model.bart_rec_prefix_model import Prefix_BartForRec
from model.bert_rec_model import MyBertForSequenceClassification
from model.gpt_rec_model import MyGPT2ForSequenceCLassification
from model.gpt_rec_prefix_model import Prefix_GPT2ForRec
from model.bart_rec_model import MyBartForSequenceClassification
# from metrics import f1_recall_precision_metric

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.9.0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

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

    # Data loading from LMRec
    if data_args.dataset_name is not None:
        with open(data_args.dataset_name, "rb") as inp:
            dataset = pickle.load(inp)
        logger.info(f"Load the dataset successfully {data_args.dataset_name}")
    elif data_args.yelp_dataset_city is not None:
        if "bert" == model_args.model_type: 
            city_map = {"toronto": "dataset/yelp_toronto_selected_bert.pkl"}
        elif "gpt2" == model_args.model_type:
            city_map = {"toronto": "dataset/yelp_toronto_selected_gpt2.pkl"}
        elif "bart" == model_args.model_type:
            city_map = {"toronto": "dataset/yelp_toronto_selected_bart.pkl"}

        with open(city_map[data_args.yelp_dataset_city], "rb") as inp:
            dataset = pickle.load(inp)
        # dataset = Dataset(
        #     data_args.yelp_dataset_city, 
        #     csv_file=city_map[data_args.yelp_dataset_city],
        #     masking=False, 
        #     model_type = model_args.model_type, 
        #     max_len = data_args.max_seq_length, 
        #     # This part is simplified since we generally takes max_seq_len = 400, 
        #     # if max_seq_len + prefix_seq_len > 512 for bert or > 1024 for gpt2
        #     # the max_len should be max_seq_len - prefix_seq_len
        # )
        # with open(city_map[data_args.yelp_dataset_city], "wb") as outp: pickle.dump(dataset, outp)
        logger.info(f"Load the dataset successfully {city_map[data_args.yelp_dataset_city]}")

    if training_args.do_train:
        logger.info(f"The input_data_mode for train is {data_args.input_data_mode}")
        if data_args.input_data_mode == "keyphrase":
            train_dataset = load_rec_dataset(
                X = dataset.X_key_train, 
                y = dataset.y_train, 
                user_labels = dataset.user_labels_train, 
                max_samples = data_args.max_train_samples
            )
        elif data_args.input_data_mode == "review":
            train_dataset = load_rec_dataset(
                X = dataset.X_train, 
                y = dataset.y_train, 
                user_labels = dataset.user_labels_train,
                max_samples = data_args.max_train_samples
            )
        else:
            raise ValueError("Please specify data_args.input_data_mode to be 'keyphrase' or 'review'")
    if training_args.do_eval:
        logger.info(f"The input_data_mode for eval is {data_args.input_data_mode}")
        if data_args.input_data_mode == "keyphrase":
            eval_dataset = load_rec_dataset(
                X = dataset.X_key_eval, 
                y = dataset.y_eval, 
                user_labels = dataset.user_labels_eval, 
                max_samples = data_args.max_eval_samples
            )
        elif data_args.input_data_mode == "review":
            eval_dataset = load_rec_dataset(
                X = dataset.X_eval, 
                y = dataset.y_eval, 
                user_labels = dataset.user_labels_eval, 
                max_samples = data_args.max_eval_samples
            )
        else:
            raise ValueError("Please specify data_args.input_data_mode to be 'keyphrase' or 'review'")
    if training_args.do_predict:
        logger.info(f"The input_data_mode for test is {data_args.input_data_mode}")
        if data_args.input_data_mode == "keyphrase":
            test_dataset = load_rec_dataset(
                X = dataset.X_key_test, 
                y = dataset.y_test, 
                user_labels = dataset.user_labels_test, 
                max_samples = data_args.max_predict_samples
            )
        elif data_args.input_data_mode == "review":
            test_dataset = load_rec_dataset(
                X = dataset.X_test, 
                y = dataset.y_test, 
                user_labels = dataset.user_labels_test, 
                max_samples = data_args.max_predict_samples
            )
        else:
            raise ValueError("Please specify data_args.input_data_mode to be 'keyphrase' or 'review'")

    # if model_args.model_name_or_path in ['gpt2', 'gpt2-medium', 'bert-base-uncased']:
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # This section is to set up some other attributes in model config
    if model_args.model_type in ['gpt2', 'bart']:
        if model_args.model_type == 'gpt2':
            config.pad_token_id = 50256
        elif model_args.model_type == "bart":
            config.pad_token_id = 1
        
        if model_args.tuning_mode == "prefixtune":
            if model_args.num_users == 0:
                raise ValueError("Please specify the num_users")
            if model_args.prefix_seq_len == 0:
                raise ValueError("Please specify prefix_seq_len")
            if model_args.mid_dim == 0:
                raise ValueError("Please specify mid_dim (512 in general)")
            if model_args.with_interaction is None:
                raise ValueError("Please specify whether the interaction layer should be added")

            config.preseqlen = model_args.prefix_seq_len
            config.num_users = model_args.num_users
            config.mid_dim = model_args.mid_dim
            config.tuning_mode = "prefixtune"
            config.with_interaction = model_args.with_interaction
        elif model_args.tuning_mode == "finetune":
            config.tuning_mode = "finetune"
        else:
            raise ValueError(f"Unrecoginized tuning_mode {model_args.tuning_mode}")

    # There are totally 1121 labels for Yelp_Toronto.csv
    if model_args.num_labels != 0:
        config.num_labels = model_args.num_labels
    elif data_args.yelp_dataset_city is not None:
        config.num_labels = len(train_dataset.labels) if training_args.do_train else len(set(eval_dataset.labels))
    else:
        raise ValueError("Please specify the number of labels in the dataset")

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=True,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )

    if model_args.model_type == "bert":
        model = MyBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        logger.info("Initialize Bert for Rec successfully")
    elif model_args.model_type == "gpt2":
        if model_args.tuning_mode == "finetune":
            model = MyGPT2ForSequenceCLassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            logger.info("Initialize GPT2 for Rec successfully")
        elif model_args.tuning_mode == 'prefixtune':
            if model_args.model_name_or_path not in ["gpt2", "gpt2-medium"]:
                model = Prefix_GPT2ForRec.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
                logger.info("Load Prefix Tuning GPT2 for Rec successfully")
            else:
                # Freeze GPT2 Parameter
                model = MyGPT2ForSequenceCLassification(config)
                logger.info("Freeze gpt2 parameters")
                for n, param in model.named_parameters():
                    if "transformer" in n:
                        param.requires_grad = False
                model = Prefix_GPT2ForRec(config, model)
                logger.info("Initialize Prefix Tuning GPT2 for Rec successfully")
        else:
            raise ValueError(f"Please specify the tuning mode other than {model_args.tuning_mode}")
    elif model_args.model_type == "bart":
        if model_args.tuning_mode == "finetune":
            # if model_args.model_name_or_path != "facebook/bart-base":
            model = MyBartForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        elif model_args.tuning_mode == "prefixtune":
            if model_args.model_name_or_path not in ['facebook/bart-base']:
                model = Prefix_BartForRec.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
                logger.info("Load Prefix Tuning Bart for Rec successfully")
            else:
                model = MyBartForSequenceClassification(config)
                # Freeze Bart Parameter
                logger.info("Freeze Bart parameters")
                for n, param in model.named_parameters():
                        if "transformer" in n:
                            param.requires_grad = False
                model = Prefix_BartForRec(config, model)
                logger.info("Initialize Prefix Tuning Bart for Rec successfully")
    else:
        raise ValueError("The model type can only be bert, gpt2 or bart")
    
    # if model_args.tuning_mode == "prefixtune":
    data_collator = DataCollatorForYelpRec(
        tuning_mode = model_args.tuning_mode, 
        prefix_seq_len = model_args.prefix_seq_len,
        model_type = model_args.model_type,
        prefix_only = model_args.prefix_only,
    ) 
    # else:
    #     data_collator = DataCollatorForYelpRec(tuning_mode = model_args.tuning_mode)

    def compute_metrics(eval_predictions: EvalPrediction):
        predictions = eval_predictions.predictions
        y_pred = softmax(predictions, axis = 1)
        y_true = eval_predictions.label_ids

        def mean_reciprocal_rank(y_true, y_score):
            one_hot = np.eye(len(y_score[0]))[y_true]
            temp = np.argsort(-1 * y_score, axis=1)
            ranks = temp.argsort() + 1
            scores = 1/np.sum(one_hot*ranks, axis=1)
            return np.mean(scores), (1.645 * np.std(scores)) / np.sqrt(len(scores))

        f1 = f1_score(y_true = y_true, y_pred = np.argmax(y_pred, axis=1), average="weighted")
        recall = recall_score(y_true = y_true, y_pred = np.argmax(y_pred, axis=1), average="weighted")
        precision = precision_score(y_true = y_true, y_pred = np.argmax(y_pred, axis=1), average="weighted")

        hit_rates5 = top_k_accuracy_score(y_true, y_pred, k=5, labels = range(model_args.num_labels))
        hit_rates10 = top_k_accuracy_score(y_true, y_pred, k=10, labels = range(model_args.num_labels))
        hit_rates20 = top_k_accuracy_score(y_true, y_pred, k=20, labels = range(model_args.num_labels))
        accuracy = top_k_accuracy_score(y_true, y_pred, k=1, labels = range(model_args.num_labels)) 
        mrr = mean_reciprocal_rank(y_true, y_pred)

        out = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
            'MRR_0': mrr[0], 'MRR_1': mrr[1], 'HR@5': hit_rates5, 'HR@10': hit_rates10, 'HR@20': hit_rates20}
        return out
    
    params = [p for p in model.parameters() if p.requires_grad]
    if model_args.tuning_mode == "prefixtune": 
        name_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"All params that will be optimized are {name_params}")
    optimizer = Adam(params=params, lr=training_args.learning_rate)
    # scheduler = StepLR(optimizer, step_size=0, gamma=0.0)
    scheduler = ConstantLR(optimizer)

    # params = [p for p in model.parameters()]
    # optimizer = Adam(params=params, lr=training_args.learning_rate)
    # # scheduler = StepLR(optimizer, step_size=0, gamma=0.0)
    # scheduler = ConstantLR(optimizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics
    )

    # Training
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
    
    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(test_dataset)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        metrics["test_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

if __name__ == "__main__":
    main()