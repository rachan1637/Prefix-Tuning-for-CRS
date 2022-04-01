import os
import pickle
import random
import time
import copy
import json
from typing import Dict, List, Optional
import ast
from numpy import False_
import torch
from torch.utils.data.dataset import Dataset

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.get_logger(__name__)

def load_dataset(X, y, max_samples, user_labels = None):
    if max_samples is not None:
        max_train_samples = max_samples
        input_ids, attention_mask, labels, user_labels = X[0][:max_train_samples], X[1][:max_train_samples], y[:max_train_samples], user_labels[:max_train_samples]
        # input_ids, attention_mask, labels = X[0][:max_train_samples], X[1][:max_train_samples], y[:max_train_samples]
    else:
        input_ids, attention_mask, labels, user_labels = X[0], X[1], y, user_labels
        # input_ids, attention_mask, labels = X[0], X[1], y
    if user_labels is not None:
        dataset = {
            "input_ids": torch.tensor(input_ids), 
            "attention_mask": torch.tensor(attention_mask), 
            "labels": torch.tensor(labels),
            "user_labels": torch.tensor(user_labels)
        }
    else:
        dataset = {
            "input_ids": torch.tensor(input_ids), 
            "attention_mask": torch.tensor(attention_mask), 
            "labels": torch.tensor(labels),
        }
    dataset = YelpRecDataset(dataset)
    return dataset


@dataclass
class YelpRecDataset:
    input_ids=None,
    attention_mask=None,
    labels=None,

    def __init__(self, input_dict):
        self.input_ids = input_dict['input_ids']
        self.attention_mask = input_dict['attention_mask']
        self.labels = input_dict['labels']
        if "user_labels" in input_dict:
            self.user_labels = input_dict['user_labels']
            self.has_user_ids = True
        else:
            self.has_user_ids = False

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        if self.has_user_ids:
            return (
                torch.tensor(self.input_ids[i], dtype=torch.long),
                torch.tensor(self.attention_mask[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.user_labels[i], dtype=torch.long)
            )
        else:
            return (
                torch.tensor(self.input_ids[i], dtype=torch.long),
                torch.tensor(self.attention_mask[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long)
            )

@dataclass
class DataCollatorForYelpRec:
    tuning_mode: None
    prefix_seq_len: 0
    model_type: None
    prefix_only: False
        
    def __call__(self, examples):
        if len(examples[0]) == 4:
            input_ids, attention_mask, labels, user_labels = zip(*examples)
        else:
            input_ids, attention_mask, labels = zip(*examples)

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim = 0)
        labels = torch.stack(labels, dim = 0)

        # In gpt2, we manually prepend the prefix_len * [1] the attention_mask
        if self.tuning_mode == "prefixtune" and self.model_type == 'gpt2' and self.prefix_seq_len > 0:
            bs = input_ids.shape[0]
            additional_att = torch.stack([torch.tensor([1] * self.prefix_seq_len)] * bs, dim = 0)
            attention_mask = torch.concat([additional_att, attention_mask], dim = 1)
        
        if self.tuning_mode == "prefixtune" and self.prefix_only:
            input_ids = None
            attention_mask = None

        if self.tuning_mode == "finetune":
            # del user_labels
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        elif self.tuning_mode == "prefixtune":
            user_labels = torch.stack(user_labels, dim = 0)

            # labels[labels == self.tokenizer.pad_token_id] = -100 # tgt
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "user_labels": user_labels}
        else:
            raise ValueError(f"Please specify tuning mode to be finetune or prefixtune other than {self.tuning_mode}")

@dataclass
class DataCollatorForMultiUserRecommendation:
    """
    Data collator used for Single User language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]

        input_ids, labels, user_ids, _ = zip(*examples)

        batch = self._tensorize_batch(input_ids)
        label_ids = torch.stack(labels, dim = 0)
        user_ids = torch.stack(user_ids, dim = 0)

        # labels[labels == self.tokenizer.pad_token_id] = -100 # tgt
        return {"input_ids": batch, "labels": label_ids, "user_ids": user_ids}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)