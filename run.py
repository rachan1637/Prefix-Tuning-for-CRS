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
LMRec
"""
import logging
import os
import re
import sys
import comet_ml
import pickle
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR
import torch
import torch.nn as nn
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
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from data import Dataset

from data_utils import LineByLineJsonRecommendationDataset, DataCollatorForYelpRec, YelpRecDataset, DataCollatorForMultiUserRecommendation

from arguments import ModelArguments, DataTrainingArguments
from model import MyBertForSequenceClassification
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
    if data_args.yelp_dataset_city is not None:
        city_map = {"toronto": "dataset/yelp_toronto.pkl"}
        with open(city_map[data_args.yelp_dataset_city], "rb") as inp:
            dataset = pickle.load(inp)
        # dataset = Dataset(data_args.yelp_dataset_city, masking=True)
        logger.info("Load the dataset successfully")

        if training_args.do_train:
            if data_args.max_train_samples is not None:
                max_train_samples = data_args.max_train_samples
                input_ids, attention_mask, labels = dataset.X_train[0][:max_train_samples], dataset.X_train[1][:max_train_samples], dataset.y_train[:max_train_samples]
            else:
                input_ids, attention_mask, labels = dataset.X_train[0], dataset.X_train[1], dataset.y_train
            train_dataset = {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask), "labels": torch.tensor(labels)}
            train_dataset = YelpRecDataset(train_dataset)
        if training_args.do_eval:
            if data_args.max_eval_samples is not None:
                max_eval_samples = data_args.max_eval_samples
                input_ids, attention_mask, labels = dataset.X_eval[0][:max_eval_samples], dataset.X_eval[1][:max_eval_samples], dataset.y_eval[:max_eval_samples]
            else:
                input_ids, attention_mask, labels = dataset.X_eval[0], dataset.X_eval[1], dataset.y_eval
            eval_dataset = {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask), "labels": torch.tensor(labels)}
            eval_dataset = YelpRecDataset(eval_dataset)
        if training_args.do_predict:
            if data_args.max_predict_samples is not None:
                max_predict_samples = data_args.max_predict_samples
                input_ids, attention_mask, labels = dataset.X_test[0][:max_predict_samples], dataset.X_test[1][:max_predict_samples], dataset.y_test[:max_predict_samples]
            else:
                input_ids, attention_mask, labels = dataset.X_test[0], dataset.X_test[1], dataset.y_test
            test_dataset = {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask), "labels": torch.tensor(labels)}
            test_dataset = YelpRecDataset(test_dataset)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # This section is to set up some other attributes in model config

    # There are totally 1121 labels for Yelp_Toronto.csv
    if model_args.num_labels != 0:
        config.num_labels = model_args.num_labels
    elif data_args.yelp_dataset_city is not None:
        config.num_labels = len(set(train_dataset.labels)) if training_args.do_train else len(set(eval_dataset.labels))
    else:
        raise ValueError("Please specify the number of labels in the dataset")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # model = MyBertForSequenceClassification.from_config(config)
    model = MyBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Get the datasets if it's not created from LMRec
    if data_args.yelp_dataset_city is None:
        if data_args.train_file is not None and training_args.do_train:
            train_dataset = LineByLineJsonRecommendationDataset(
                tokenizer=tokenizer,
                file_path=data_args.train_file,
                block_size=max_seq_length,
                input_data_mode="review",
                model_type=model_args.model_type
            )
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset[:data_args.max_train_samples]
        if data_args.validation_file is not None and training_args.do_eval:
            eval_dataset = LineByLineJsonRecommendationDataset(
                tokenizer=tokenizer,
                file_path=data_args.validation_file,
                block_size=max_seq_length,
                input_data_mode="review",
                model_type=model_args.model_type
            )
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset[:data_args.max_eval_samples]
        if data_args.test_file is not None and training_args.do_predict:
            test_dataset = LineByLineJsonRecommendationDataset(
                tokenizer=tokenizer,
                file_path=data_args.test_file,
                block_size=max_seq_length,
                input_data_mode="review",
                model_type=model_args.model_type
            )
            if data_args.max_test_samples is not None:
                test_dataset = test_dataset[:data_args.max_test_samples]
    
    data_collator = DataCollatorForYelpRec() if data_args.yelp_dataset_city is not None else DataCollatorForMultiUserRecommendation(tokenizer)

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

        hit_rates5 = top_k_accuracy_score(y_true, y_pred, k=5)
        hit_rates10 = top_k_accuracy_score(y_true, y_pred, k=10)
        hit_rates20 = top_k_accuracy_score(y_true, y_pred, k=20)
        accuracy = top_k_accuracy_score(y_true, y_pred, k=1)
        mrr = mean_reciprocal_rank(y_true, y_pred)

        out = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
            'MRR_0': mrr[0], 'MRR_1': mrr[1], 'HR@5': hit_rates5, 'HR@10': hit_rates10, 'HR@20': hit_rates20}
        return out
    
    params = [p for p in model.parameters()]
    optimizer = Adam(params=params, lr=training_args.learning_rate)
    # scheduler = StepLR(optimizer, step_size=0, gamma=0.0)
    scheduler = ConstantLR(optimizer)

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