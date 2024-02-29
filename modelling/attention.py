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
    

# improved version of Attention class (already tested and working correctly) over x5 speedup compared to above
# even quicker is attention = F.scaled_dot_product_attention(q, k, v, mask) # x1.4 quicker than below 
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

        #attn_weight = torch.baddbmm(mask, q, k.transpose(-2, -1), beta=scale_factor, alpha=scale_factor)
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
        self.query_transform = nn.Linear(d_model, d_model, bias=bias)
        self.key_transform = nn.Linear(d_model, d_model, bias=bias)
        self.value_transform = nn.Linear(d_model, d_model, bias=bias)
        self.output_transform = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, attention_mask=None):
        # input shape of q, k, v: (batch_size, seq_len, d_model)
        # Diff to og: apply weight matrix on whole input but split into n_heads so that each head has own part of weight matrix
        q = (
            self.query_transform(q)
            .view(q.size(0), q.size(1), self.n_heads, self.d_model//self.n_heads)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.key_transform(k)
            .view(k.size(0), k.size(1), self.n_heads, self.d_model//self.n_heads)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.value_transform(v)
            .view(v.size(0), v.size(1), self.n_heads, self.d_model//self.n_heads)
            .transpose(1, 2)
            .contiguous()
        )

        # q, k, v shape: (batch_size, n_heads, seq_len, d_model/n_heads)
        #q = q.view(-1, q.size(2), q.size(3))
        #k = k.view(-1, k.size(2), k.size(3))
        #v = v.view(-1, v.size(2), v.size(3))

        # q, k, v shape: (batch_size, n_heads, seq_len, d_model/n_heads)
        attention = self.attention(q, k, v, attention_mask.unsqueeze(1))

        # reverse view operations (concatenation of heads)
        """attention = attention.view(
            -1, self.n_heads, attention.size(1), attention.size(2)
        )"""
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(attention.size(0), attention.size(1), self.d_model)

        return self.output_transform(attention)