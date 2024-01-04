import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
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
        mask = mask.masked_fill(attention_mask.unsqueeze(1) == 0, float(-1e10))

        attention = torch.matmul(
            F.softmax(
                torch.matmul(q, k.transpose(1, 2)) / np.sqrt(q.size(-1)) + mask,
                dim=-1,
            ),
            v,
        )
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask_future=False, bias=False):
        super().__init__()
        self.attention = Attention(mask_future)
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, attention_mask=None):
        # input shape of q, k, v: (batch_size, seq_len, d_model)
        # Diff to og: apply weight matrix on whole input but split into n_heads so that each head has own part of weight matrix
        q = self.W_q(q).view(q.size(0), q.size(1), self.n_heads, -1).transpose(1, 2)
        k = self.W_k(k).view(k.size(0), k.size(1), self.n_heads, -1).transpose(1, 2)
        v = self.W_v(v).view(v.size(0), v.size(1), self.n_heads, -1).transpose(1, 2)

        # q, k, v shape: (batch_size, n_heads, seq_len, d_model/n_heads)
        q = q.view(-1, q.size(2), q.size(3))
        k = k.view(-1, k.size(2), k.size(3))
        v = v.view(-1, v.size(2), v.size(3))

        # q, k, v shape: (batch_size*n_heads, seq_len, d_model/n_heads)
        attention = self.attention(q, k, v, attention_mask)

        # reverse view operations
        attention = attention.view(
            -1, self.n_heads, attention.size(1), attention.size(2)
        )
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(attention.size(0), attention.size(1), -1)

        return self.W_o(attention)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x):
        return self.ffn(x)


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # include dropout after the multi-head attention and position wise feed forward layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))  # residual connection


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward,
        dropout=0.1,
        mask_future=False,
        bias=False,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            d_model, n_heads, mask_future, bias
        )
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.position_wise_ffn = PositionWiseFFN(d_model, dim_feedforward)
        self.add_norm_2 = AddNorm(d_model, dropout)

    def forward(self, x, attention_mask=None):
        # input shape of x: (batch_size, seq_len, d_model)
        x = self.add_norm_1(x, self.multi_head_attention(x, x, x, attention_mask))
        x = self.add_norm_2(x, self.position_wise_ffn(x))
        return x
