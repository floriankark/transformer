import math
import torch
import torch.nn as nn
from typing import Optional

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 attn_mask: Optional[torch.Tensor] = None, mask_future: bool = False) -> torch.Tensor:
    L, S = q.size(-2), k.size(-2)
    scale_factor = 1/math.sqrt(k.size(-1))

    attn_bias = torch.zeros(L, S, dtype=q.dtype).to(q.device)
    if mask_future:
        # if fp16, then fill with -1e4, or else value cannot be converted to type at::Half without overflow
        # same goes for float("-inf") -> leads to nan loss
        attn_bias = torch.triu(torch.full((L, S), float(-1e4), dtype=q.dtype, device=q.device), diagonal=1)

    # TODO: check if it is more stable if q and k are scaled individually
    attn_weight = (q @ k.transpose(-2, -1)) * scale_factor
    attn_weight = attn_weight + attn_bias # broadcasting

    # here attn_mask is the key padding mask
    if attn_mask is not None:
        attn_weight = attn_weight.masked_fill(attn_mask == 0, float(-1e4))

    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False) # true in some implementations

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    """def _reset_parameters(self):
    # assumption that q, k have mean 0 and variance 1 (= std 1), see 3.2.1 in the paper
        nn.init.normal_(self.q_proj, mean=0, std=1)
        nn.init.normal_(self.k_proj, mean=0, std=1)
        # TODO: check if below works or if it should be normal_
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)"""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, mask_future: bool = False) -> torch.Tensor:
        # q, k, v: (batch_size, seq_len, d_model)
        batch_size = q.size(0)
        seq_len = q.size(1)
        q = (self.q_proj(q).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2))
        k = (self.k_proj(k).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2))
        v = (self.v_proj(v).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2))
        # q, k, v: (batch_size, n_heads, seq_len, d_model/n_heads)

        # attn_mask: (batch_size, seq_len)
        attn_mask = attn_mask.to(q.dtype)
        attn_mask = attn_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.n_heads, seq_len, -1)
        # attn_mask: (batch_size, n_heads, seq_len, seq_len) -> to match shape of qk^T

        attn_output = scaled_dot_product_attention(q, k, v, attn_mask, mask_future=mask_future)
        
        # attn_output: (batch_size, n_heads, seq_len, d_model/n_heads)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output: (batch_size, seq_len, n_heads, d_model/n_heads)
        attn_output = attn_output.view(batch_size, seq_len, self.d_model) # concatenate
        # attn_output: (batch_size, seq_len, d_model)
        return self.out_proj(attn_output)