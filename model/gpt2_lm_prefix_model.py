from random import sample
import torch
from transformers import GPT2PreTrainedModel
from torch import  nn

class PrefixTuning_GPT2ForLM(GPT2PreTrainedModel):
    """Prefix tuning for GPT2 LM model"""
    def __init__(self, config, gpt2_model):
        super().__init__(config)
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # Set up from config
        self.preseqlen = config.preseqlen
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.mid_dim = config.mid_dim
        self.with_interaction = config.with_interaction
        self.add_item_prefix = config.add_item_prefix

        if self.with_interaction:
            self.wte_user = nn.Embedding(self.preseqlen * self.num_users, config.n_embd)
            self.control_trans_user = nn.Sequential(
                nn.Linear(config.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd)
            )

            if self.add_item_prefix:
                self.wte_item = nn.Embedding(self.preseqlen * self.num_items, config.n_embd)
                self.control_trans_item = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd)
                )
        else:
            self.wte_user = nn.Embedding(self.preseqlen * self.num_users, config.n_layer * 2 * config.n_embd)

            if self.add_item_prefix:
                self.wte_item = nn.Embedding(self.preseqlen * self.num_items, config.n_layer * 2 * config.n_embd)
        
        # Here we set prefix dropout prob = 0.4
        self.prefix_dropout = 0.4
        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))

        self.model_parallel = False
        self.device_map = None

        self.gpt2 = gpt2_model


    def get_prompt(self, user_labels, item_labels = None, bsz=None, sample_size = 1):
        # bsz = bsz * sample_size
        if user_labels is not None:
            input_tokens_user = torch.stack(
                [torch.arange(user_label, user_label + self.preseqlen).long() for user_label in user_labels for _ in range(sample_size)], dim = 0
            ).to(self.device)
        else:
            input_tokens_user = torch.arange(self.preseqlen).long()
            input_tokens_user = input_tokens_user.unsqueeze(0).expand(bsz, -1).to(self.device)

        if self.add_item_prefix:
            input_tokens_item = torch.stack(
                [torch.arange(item_label, item_label + self.preseqlen).long() for item_label in item_labels for _ in range(sample_size)], dim = 0
            ).to(self.device)

        # input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        if self.with_interaction:
            temp_control_user = self.wte_user(input_tokens_user)
            past_key_values_user = self.control_trans_user(temp_control_user) #bsz, seqlen, layer*emb

            if self.add_item_prefix:
                temp_control_item = self.wte_item(input_tokens_item)
                past_key_values_item = self.control_trans_item(temp_control_item)
                past_key_values = torch.concat([past_key_values_user, past_key_values_item], dim = 1)
                assert past_key_values.shape[1] == self.preseqlen * 2
            else:
                past_key_values = past_key_values_user
        else:
            past_key_values_user = self.wte_user(input_tokens_user)

            if self.add_item_prefix:
                past_key_values_item = self.wte_item(input_tokens_item)
                past_key_values = torch.concat([past_key_values_user, past_key_values_item], dim = 1)
                assert past_key_values.shape[1] == self.preseqlen * 2
            else:
                past_key_values = past_key_values_user

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for _, key_val in enumerate(past_key_values):
            result.append(
                {
                    "prev_key": key_val[0].contiguous(),
                    "prev_value": key_val[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() # bsz, preseqlen
                }
            )
        return result

    def forward(
        self,
        input_ids,
        attention_mask,
        user_labels,
        item_labels=None,
        **kwargs,
    ):
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(user_labels, item_labels, bsz=bsz)

        output = self.gpt2(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            past_prompt = past_prompt, 
            **kwargs
        )

        return output


    def generate(self,
                 input_ids,
                 attention_mask,
                 user_labels,
                 item_labels=None,
                 **kwargs):

        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            user_labels, item_labels, bsz=bsz, sample_size=kwargs['num_beams'],
        )
        generated_ids = self.gpt2.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids

    