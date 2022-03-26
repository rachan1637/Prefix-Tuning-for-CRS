
from sklearn.datasets import load_digits
from transformers import GPT2PreTrainedModel
import torch.nn as nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

class MyGPT2ForSequenceCLassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        if config.tuning_mode == "finetune":
            from transformers import GPT2Model
            self.transformer = GPT2Model(config)
        elif config.tuning_mode == "prefixtune":
            from modeling_gpt2 import GPT2Model
            self.transformer = GPT2Model(config)
        else:
            raise ValueError(f"Unrecognized tuning mode {config.tuning_mode}")

        self.dropout = nn.Dropout(0.4)

        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=config.n_embd, out_features=1024, bias=True),
            nn.ReLU(),
        )

        self.classifier2 = nn.Linear(in_features=1024, out_features=self.config.num_labels, bias=True)

        self.model_parallel = False
        self.device_map = None

        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        user_labels = None,
        past_key_values = None,
        **kwargs,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values = past_key_values,
            attention_mask = attention_mask,
            **kwargs,
        )

        hidden_state = transformer_outputs[0]
        logits = self.classifier1(hidden_state)
        logits = self.classifier2(logits)

        batch_size, sequence_length = input_ids.shape[:2]
        
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        
        pooled_logits = logits[torch.arange(batch_size, device = self.device), sequence_lengths]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
        )