import torch.nn as nn

from transformers import (
    AutoModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # print(config)
        self.config = config

        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.4)

        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=config.hidden_size, out_features=1024, bias=True),
            nn.ReLU(),
        )

        self.classifier2 = nn.Linear(in_features=1024, out_features=self.config.num_labels, bias=True)

        self.init_weights()

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        out = outputs['pooler_output']

        logits = self.classifier1(out)
        logits = self.dropout(logits)
        logits = self.classifier2(logits)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits.view(-1, self.config.num_labels),
        )