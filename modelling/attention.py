import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


"""class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, q, k, v, attention_mask=None):
        # input shape of q, k, v: (batch_size, seq_len, d_model)
        mask = torch.zeros(q.size(1), k.size(1))
        if self.mask_future:
            mask = mask.masked_fill(
                torch.triu(torch.ones(mask.shape), diagonal=1) == 1, float(-1e10)
            )

        mask = torch.stack([mask] * q.size(0))

        repetition = int(
            q.size(0) / attention_mask.size(0)
        )  # basically number of heads

        if attention_mask is not None:
            attention_mask = (
                attention_mask.unsqueeze(-1)  # (batch_size, 1, key seq_len)
                .transpose(1, 2)  # (batch_size, key seq_len, 1)
                .repeat_interleave(
                    repetition, dim=0
                )  # (batch_size*n_heads, key seq_len, 1)
                .expand_as(mask)  # (batch_size*n_heads, key seq_len, query seq_len)
            )

        mask = mask.to(q.device)
        mask = mask.masked_fill(attention_mask == 0, float(-1e10))

        attention = torch.matmul(
            F.softmax(
                torch.matmul(q, k.transpose(1, 2)) / np.sqrt(q.size(-1)) + mask,
                dim=-1,
            ),
            v,
        )
        return attention"""
    

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



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_future=False, bias=False):
        super().__init__()
        self.attention = Attention(mask_future)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_transform = nn.Linear(d_model, d_model, bias=bias)
        self.key_transform = nn.Linear(d_model, d_model, bias=bias)
        self.value_transform = nn.Linear(d_model, d_model, bias=bias)
        self.output_transform = nn.Linear(d_model, d_model, bias=bias)

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
        attention = self.attention(q, k, v, attention_mask.unsqueeze(1))
        
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, self.d_model)

        return self.output_transform(attention)