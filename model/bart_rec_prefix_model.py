
from transformers import PretrainedBartModel

import torch.nn as nn
import torch

class Prefix_BartForRec(PretrainedBartModel):
    """Prefix tuning for bart classification model"""
    def __init__(self, config):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        # Set up from config
        self.preseqlen = config.preseqlen
        self.num_users = config.num_users
        self.mid_dim = config.mid_dim

        self.prefix_dropout = 0.4
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.wte = nn.Embedding(self.preseqlen * self.num_users, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        # encoder prefix
        self.wte_enc = nn.Embedding(self.preseqlen * self.num_users, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # cross prefix
        self.wte2 = nn.Embedding(self.preseqlen * self.num_users, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))

    def set_bart(self, bart_model):
        self.bart = bart_model
        print("GPT2 is set")
    
    def get_prompt(self, user_labels, bsz=None, sample_size=1):
        # Sample size is used when decoding strategy is beam decoding
        old_bsz = bsz
        bsz = bsz * sample_size

        past_key_values, _, seqlen = self.get_past_key_values(bsz, self.wte, self.control_trans, user_labels)
        past_key_values_cross, _, _ = self.get_past_key_values(bsz, self.wte2, self.control_trans2, user_labels)
        past_key_values_encoder, bsz_enc, _ = self.get_past_key_values(old_bsz, self.wte_enc, self.control_trans_enc, user_labels)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {
                "decoder_prompt": {
                    "prev_key": key_val[0].contiguous(),
                    "prev_value": key_val[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() # bsz, preseqlen
                }
            }

            key_val2 = past_key_values_cross[i]
            temp_dict['cross_attention_prompt'] = {
                "prev_key": key_val2[0].contiguous(),
                "prev_value": key_val2[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
            }

            key_val_enc = past_key_values_encoder[i]
            temp_dict['encoder_prompt'] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
            }
            result.append(temp_dict)

        return result

    def get_past_key_values(self, bsz, wte, control_trans, user_labels):
        if self.num_users != 1:
            input_tokens = torch.stack(
                [torch.arange(user_label, user_label + self.preseqlen).long() for user_label in user_labels], dim = 0
            ).to(self.device)
        else:
            input_tokens = torch.arange(self.preseqlen).long()
            input_tokens = input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        
        temp_control = wte(input_tokens)
        past_key_values = control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values, bsz, seqlen

    def forward(
        self,
        input_ids,
        attention_mask,
        user_labels,
        **kwargs,
    ):
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(user_labels, bsz=bsz)

        output = self.bart(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            past_prompt = past_prompt, 
            **kwargs
        )

        return output
        

