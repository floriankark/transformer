import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# even quicker is attention = F.scaled_dot_product_attention(q, k, v, mask)
class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, q, k, v, attention_mask=None):
        L, S = q.size(-2), k.size(-2)
        scale_factor = 1/math.sqrt(q.size(-1))
        mask = torch.zeros(L, S, dtype=q.dtype)

        if self.mask_future:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            mask = mask.masked_fill(temp_mask.logical_not(), float("-inf"))
            mask.unsqueeze_(0)
        
            if q.dim() == 4:
                mask.unsqueeze_(1)

        mask = mask.to(q.device)
        attn_weight = (q @ k.transpose(-2, -1) + mask) * scale_factor

        if attention_mask is not None:
            attn_bias = attention_mask.unsqueeze(1)
            attn_bias = attn_bias.to(q.device)
            attn_weight = attn_weight.masked_fill(attn_bias == 0, float("-inf"))

        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight @ v
            
        return attn_weight
    
def scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    attn_bias = attn_bias.to(query.device)
    attn_weight = (query @ key.transpose(-2, -1)) * scale_factor
    attn_weight = attn_weight + attn_bias

    if is_causal:
        temp_mask = temp_mask.to(query.device)
        attn_weight = attn_weight.masked_fill(temp_mask.logical_not(), 0)

    if attn_mask is not None:
        attn_mask = attn_mask.to(query.dtype)
        attn_weight = attn_weight.masked_fill(attn_mask == 0, float("-inf"))

    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_future=False):
        super().__init__()
        self.mask_future = mask_future
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_transform.weight)
        nn.init.xavier_uniform_(self.key_transform.weight)
        nn.init.xavier_uniform_(self.value_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)

    def forward(self, q, k, v, attention_mask=None):
        # input shape of q, k, v: (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        seq_len = q.size(1)
        q = (
            self.query_transform(q)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.key_transform(k)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.value_transform(v)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # q, k, v shape: (batch_size, n_heads, seq_len, d_model/n_heads)
        #attention = self.attention(q, k, v, attention_mask.unsqueeze(1))
        # give attention mask d type of q
        attention_mask = attention_mask.to(q.dtype)
        attention = scaled_dot_product_attention(q, k, v, attention_mask.unsqueeze(1).unsqueeze(1), is_causal=self.mask_future)
        
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, self.d_model)

        return self.output_transform(attention)