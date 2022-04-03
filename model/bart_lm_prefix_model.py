from transformers import PretrainedBartModel

import torch.nn as nn
import torch

class PrefixTuning_BartforLM(PretrainedBartModel):
    def __init__(self, config, bart_model):
        super().__init__(config)
        print('under the PrefixTuning model')

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        # Set up from config
        self.preseqlen = config.preseqlen
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.mid_dim = config.mid_dim
        self.with_interaction = config.with_interaction
        self.add_item_prefix = config.add_item_prefix

        self.prefix_dropout = 0.4
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.bart = bart_model

        if self.with_interaction:
            # User prefix
            self.wte_user = nn.Embedding(self.preseqlen * self.num_users, self.n_embd)
            self.control_trans_user = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
            )

            # encoder prefix
            self.wte_enc_user = nn.Embedding(self.preseqlen * self.num_users, self.n_embd)
            self.control_trans_enc_user = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

            # cross prefix
            self.wte2_user = nn.Embedding(self.preseqlen * self.num_users, self.n_embd)
            self.control_trans2_user = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
            
            if self.add_item_prefix:
                # Item prefix
                self.wte_item = nn.Embedding(self.preseqlen * self.num_items, self.n_embd)
                self.control_trans_item = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
                )

                # encoder prefix
                self.wte_enc_item = nn.Embedding(self.preseqlen * self.num_items, self.n_embd)
                self.control_trans_enc_item = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                # cross prefix
                self.wte2_item = nn.Embedding(self.preseqlen * self.num_items, self.n_embd)
                self.control_trans2_item = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
            
        else:
            self.wte_user = nn.Embedding(self.preseqlen * self.num_users, self.match_n_layer * 2 * self.n_embd)
            self.wte_enc_user = nn.Embedding(self.preseqlen * self.num_users, self.match_n_layer * 2 * self.n_embd)
            self.wte2_user = nn.Embedding(self.preseqlen * self.num_users, self.match_n_layer * 2 * self.n_embd)

            if self.add_item_prefix:
                self.wte_item = nn.Embedding(self.preseqlen * self.num_items, self.match_n_layer * 2 * self.n_embd)
                self.wte_enc_item = nn.Embedding(self.preseqlen * self.num_items, self.match_n_layer * 2 * self.n_embd)
                self.wte2_item = nn.Embedding(self.preseqlen * self.num_items, self.match_n_layer * 2 * self.n_embd)


        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        print('total param is {}'.format(total_param))

        self.bart = bart_model


    def get_prompt(self, user_labels, item_labels, bsz=None, sample_size=1):
        # Sample size is used when decoding strategy is beam decoding
        old_bsz = bsz
        bsz = bsz * sample_size

        if self.with_interaction:
            if self.add_item_prefix:
                past_key_values, _, seqlen = self.get_past_key_values(
                    bsz, self.wte_user, user_labels, self.wte_item, item_labels, control_trans_user=self.control_trans_user, control_trans_item=self.control_trans_item
                )
                past_key_values_cross, _, _ = self.get_past_key_values(
                    bsz, self.wte2_user, user_labels, self.wte2_item, item_labels, control_trans_user=self.control_trans2_user, control_trans_item=self.control_trans2_item
                )
                past_key_values_encoder, bsz_enc, _ = self.get_past_key_values(
                    old_bsz, self.wte_enc_user, user_labels, self.wte_enc_item, item_labels, control_trans_user=self.control_trans_enc_user, control_trans_item=self.control_trans_enc_item
                )
            else:
                past_key_values, _, seqlen = self.get_past_key_values(
                    bsz, self.wte_user, user_labels, wte_item=None, item_labels=None, control_trans_user=self.control_trans_user, control_trans_item=None
                )
                past_key_values_cross, _, _ = self.get_past_key_values(
                    bsz, self.wte2_user, user_labels, wte_item=None, item_labels=None, control_trans_user=self.control_trans2_user, control_trans_item=None
                )
                past_key_values_encoder, bsz_enc, _ = self.get_past_key_values(
                    old_bsz, self.wte_enc_user, user_labels, wte_item=None, item_labels=None, control_trans_user=self.control_trans_enc_user, control_trans_item=None
                )
        else:
            if self.add_item_prefix:
                past_key_values, _, seqlen = self.get_past_key_values(
                    bsz, self.wte_user, user_labels, self.wte_item, item_labels, control_trans_user=None, control_trans_item=None
                )
                past_key_values_cross, _, _ = self.get_past_key_values(
                    bsz, self.wte2_user, user_labels, self.wte2_item, item_labels, control_trans_user=None, control_trans_item=None
                )
                past_key_values_encoder, bsz_enc, _ = self.get_past_key_values(
                    old_bsz, self.wte_enc_user, user_labels, self.wte_enc_item, item_labels, control_trans_user=None, control_trans_item=None
                )
            else:
                past_key_values, _, seqlen = self.get_past_key_values(
                    bsz, self.wte_user, user_labels, wte_item=None, item_labels=None, control_trans_user=None, control_trans_item=None
                )
                past_key_values_cross, _, _ = self.get_past_key_values(
                    bsz, self.wte2_user, user_labels, wte_item=None, item_labels=None, control_trans_user=None, control_trans_item=None
                )
                past_key_values_encoder, bsz_enc, _ = self.get_past_key_values(
                    old_bsz, self.wte_enc_user, user_labels, wte_item=None, item_labels=None, control_trans_user=None, control_trans_item=None
                )


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

    def get_past_key_values(self, bsz, wte_user, user_labels, wte_item=None, item_labels=None, control_trans_user=None, control_trans_item=None):
        if user_labels is not None:
            input_tokens = torch.stack(
                [torch.arange(user_label, user_label + self.preseqlen).long() for user_label in user_labels], dim = 0
            ).to(self.device)
        else:
            input_tokens = torch.arange(self.preseqlen).long()
            input_tokens = input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)

        if self.add_item_prefix:
            input_tokens_item = torch.stack(
                [torch.arange(item_label, item_label + self.preseqlen).long() for item_label in item_labels], dim = 0
            ).to(self.device)
        
        if self.with_interaction:
            temp_control_user = wte_user(input_tokens)
            past_key_values_user = control_trans_user(temp_control_user) #bsz, seqlen, layer*emb

            if self.add_item_prefix:
                temp_control_item = wte_item(input_tokens_item)
                past_key_values_item = control_trans_item(temp_control_item)

                past_key_values = torch.concat([past_key_values_user, past_key_values_item], dim = 1)
                assert past_key_values.shape[1] == self.preseqlen * 2
            else:
                past_key_values = past_key_values_user
        else:
            past_key_values_user = wte_user(input_tokens)

            if self.add_item_prefix:
                past_key_values_item = wte_item(input_tokens_item)
                past_key_values = torch.concat([past_key_values_user, past_key_values_item], dim = 1)
                assert past_key_values.shape[1] == self.preseqlen * 2
            else:
                past_key_values = past_key_values_user

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
        item_labels=None,
        **kwargs,
    ):
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(user_labels, item_labels, bsz=bsz)

        output = self.bart(
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
        generated_ids = self.bart.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids