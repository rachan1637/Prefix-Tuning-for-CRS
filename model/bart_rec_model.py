from transformers import (
    BartPretrainedModel, 
    BartConfig, 
    BartModel, 
)

from transformers.modeling_outputs import SequenceClassifierOutput

import torch
import torch.nn as nn

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class MyBartForSequenceClassification(BartPretrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            input_dim = config.d_model,
            inner_dim = 1024,
            num_classes = config.num_labels,
            pooler_dropout = config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        past_key_values = None,
        **kwargs
    ):

        outputs = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            past_key_values=past_key_values,
            **kwargs
        )

        hidden_states = outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]

        logits = self.classification_head(sentence_representation)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss = loss,
            logits = logits
        )
