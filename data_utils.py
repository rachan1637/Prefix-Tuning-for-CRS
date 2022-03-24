import os
import pickle
import random
import time
import copy
import json
from typing import Dict, List, Optional
import ast
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

class LineByLineJsonRecommendationDataset(Dataset):
    """
    This is created for multi-user based prefix recommendation dataset
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, input_data_mode, model_type):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, 'r', encoding = 'utf-8') as f:
            examples = json.load(f)['data']

        edited_sents = []
        user_ids = []
        label_ids = []
        for example in tqdm(examples):
            single_review_key = []
            if input_data_mode == 'keyphrase':
                for single_keypharse in example['keypharse']:
                    keypharse = single_keypharse[1]
                    single_review_key.append(keypharse)

                if len(single_review_key) == 0:
                    continue

                sent = ' | '.join(single_review_key) 
                edited_sents.append(sent)
                user_ids.append(example['user_id'])
                label_ids.append(example['business_index'])
                
            elif input_data_mode == 'review':
                review_text = example['review_text'].replace(u'\xa0', u' ')
                edited_sents.append(review_text)
                user_ids.append(example['user_id'])
                label_ids.append(example['business_index'])

        self.edited_sents = edited_sents

        if model_type == 'bert':
            batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False, padding="max_length")
        else:
            batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                    is_split_into_words=False)

        self.examples = batch_encoding["input_ids"]
        self.user_ids = user_ids
        self.label_ids = label_ids
    
        print(f"label_id: \n {self.label_ids[0]}")
        print(f"input_ids: \n {self.examples[0]}")
        print(f"keypharse sent: \n {self.edited_sents[0]}")
        print(f"use_id: \n {self.user_ids[0]}")
        # assert len(self.tgt_sent) == len(self.examples)
    
    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.label_ids[i], dtype=torch.long),
                torch.tensor(self.user_ids[i], dtype=torch.long),
                self.edited_sents[i]
                )

@dataclass
class YelpRecDataset:
    input_ids=None,
    attention_mask=None,
    label_ids=None,

    def __init__(self, input_dict):
        self.input_ids = input_dict['input_ids']
        self.attention_mask = input_dict['attention_mask']
        self.label_ids = input_dict['label_ids']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return (
            torch.tensor(self.input_ids[i], dtype=torch.long),
            torch.tensor(self.attention_mask[i], dtype=torch.long),
            torch.tensor(self.label_ids[i], dtype=torch.long)
        )

class DataCollatorForYelpRec:
    def __call__(self, examples):
        input_ids, attention_mask, label_ids = zip(*examples)

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim = 0)
        label_ids = torch.stack(label_ids, dim = 0)

        # labels[labels == self.tokenizer.pad_token_id] = -100 # tgt
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label_ids": label_ids}

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