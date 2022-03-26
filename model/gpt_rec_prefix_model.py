import torch
from transformers import GPT2PreTrainedModel
from torch import  nn

class Prefix_GPT2ForRec(GPT2PreTrainedModel):
    """Prefix tuning for GPT2 classification model"""
    def __init__(self, config):
        super().__init__(config)
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        self.preseqlen = config.preseqlen
        self.num_users = config.num_users

        self.mid_dim = config.mid_dim

        if True:
            print('[Full prefix-tuning Setting :) ]')
            # self.input_tokens = torch.arange(self.preseqlen).long()
            self.wte = nn.Embedding(self.preseqlen * self.num_users, config.n_embd)
            self.control_trans = nn.Sequential(
                nn.Linear(config.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
        
        # Here we set prefix dropout prob = 0.4
        self.prefix_dropout = 0.4
        self.dropout = nn.Dropout(self.prefix_dropout)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            # print(param.shape)
            total_param += param.numel()
        # print('total param is {}'.format(total_param))

        self.model_parallel = False
        self.device_map = None

    def set_gpt2(self, gpt2_model):
        self.gpt2 = gpt2_model
        # Freeze the gpt2 params
        for n, param in self.gpt2.named_parameters():
            if "transformer" in n:
                param.requires_grad = False
        print("GPT2 is set")

    def get_prompt(self, user_labels, bsz=None):
        if self.num_users != 1:
            input_tokens = torch.stack([torch.arange(user_label, user_label + self.preseqlen).long() for user_label in user_labels], dim = 0).to(self.device)
        else:
            input_tokens = torch.arange(self.preseqlen).long()
            input_tokens = input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        # input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,
        input_ids=None,
        user_labels=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        past_key_values_prompt = self.get_prompt(user_labels, bsz=bsz)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        # print(len(past_key_values))
        # print(past_key_values[0].shape)
        output = self.gpt2(input_ids=input_ids,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output