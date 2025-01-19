from math import sqrt
import torch
import torch.nn as nn
from typing import Optional

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 attn_mask: Optional[torch.Tensor] = None, mask_future: bool = False, 
                                 dropout: float = 0.1) -> torch.Tensor:
    L, S = q.size(-2), k.size(-2)
    scale_factor = 1.0/sqrt(k.size(-1))

    attn_bias = torch.zeros(L, S, dtype=q.dtype).to(q.device)
    if mask_future:
        # fp16 has numerical range of [-2e24, 65504]
        # if fp16, then fill with -1e4, or else value cannot be converted to type at::Half without overflow
        # same goes for float("-inf") -> leads to nan loss
        # https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/
        attn_bias = torch.triu(torch.full((L, S), float(-1e4), dtype=q.dtype, device=q.device), diagonal=1)

    # TODO: check if it is more stable if q and k are scaled individually
    attn_weight = (q @ k.transpose(-2, -1)) * scale_factor
    attn_weight = attn_weight + attn_bias # broadcasting

    # here attn_mask is the key padding mask
    if attn_mask is not None:
        attn_weight = attn_weight.masked_fill(attn_mask == 0, float(-1e4))

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, p=dropout)
    return attn_weight @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)

        """self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)"""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, mask_future: bool = False) -> torch.Tensor:
        # q, k, v: (batch_size, seq_len, d_model)
        batch_size = k.size(0)
        seq_len = k.size(1)
        seq_len_q = q.size(1)
        q = (self.q_proj(q).view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2))
        k = (self.k_proj(k).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2))
        v = (self.v_proj(v).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2))
        # q, k, v: (batch_size, n_heads, seq_len, d_model/n_heads)

        # attn_mask: (batch_size, seq_len)
        #attn_mask = attn_mask.to(q.dtype)
        attn_mask = attn_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.n_heads, seq_len, -1)
        # attn_mask: (batch_size, n_heads, seq_len, seq_len) -> to match shape of qk^T

        attn_output = scaled_dot_product_attention(q, k, v, attn_mask, mask_future=mask_future, dropout=self.attn_drop)
        
        # attn_output: (batch_size, n_heads, seq_len, d_model/n_heads)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output: (batch_size, seq_len, n_heads, d_model/n_heads)
        attn_output = attn_output.view(batch_size, seq_len, self.d_model) # concatenate
        # attn_output: (batch_size, seq_len, d_model)
        return self.resid_drop(self.out_proj(attn_output))
    

"""
Improved code
Idea: 
1. use permute and transpose where possible -> more robust
2. use 3 dim multihead attention -> faster
3. use one qkv projection -> simplified
4. more dropout for better generalization
"""

def stable_scaled_dot_product_attention2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 attn_mask: Optional[torch.Tensor] = None, mask_future: bool = False, 
                                 dropout: float = 0.1) -> torch.Tensor:
    L, S = q.size(-2), k.size(-2)
    scale_factor = 1/sqrt(sqrt(k.size(-1)))

    attn_bias = torch.zeros(L, S, dtype=q.dtype).to(q.device)
    if mask_future:
        # fp16 has numerical range of [-2e24, 65504]
        # if fp16, then fill with -1e4, or else value cannot be converted to type at::Half without overflow
        # same goes for float("-inf") -> leads to nan loss
        # https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/
        attn_bias = torch.triu(torch.full((L, S), float(-1e4), dtype=q.dtype, device=q.device), diagonal=1)

    # numerically more stable if q and k are scaled individually, https://pytorch.org/blog/accelerating-large-language-models/#appendix-a-analyzing-attention-numeric-stability
    # math: n * (A dot B) == (sqrt(n) * A) dot (sqrt(n) * B)
    # thus scale_factor = sqrt((1/sqrt(k.size(-1))) = 1/sqrt(sqrt(k.size(-1)))
    q = q * scale_factor
    attn_weight = q @ (k.transpose(-2, -1) * scale_factor)
    attn_weight = attn_weight + attn_bias # broadcasting

    # here attn_mask is the key padding mask
    if attn_mask is not None:
        attn_weight = attn_weight.masked_fill(attn_mask == 0, float(-1e4))

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, p=dropout, train=True)
    return attn_weight @ v


class RobustMultiHeadAttention(nn.Module):
    """
    Robust Multi Head Attention:
    Manipulation of batch-size to allow for different batch size between the 
    two inputs, with broadcasting over to the higher batch size value.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f'd_model ({d_model}) must be divisible by n_heads ({n_heads}) without remainder'
        self.n_heads = n_heads

        # key, query, value projections for all heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # regularization
        self.resid_drop = nn.Dropout(dropout)
        self.attn_drop = dropout

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, mask_future: bool = False) -> torch.Tensor:
        # The differentiation between q and k sizes is for the case of encoder-decoder cross attention
        # should q and k be of different sizes, then the batch size is determined by the larger of the two
        Bk, Tk, Ck = k.size() # batch size, key sequence length, key channels
        Bq, Tq, Cq = q.size() # batch size, query sequence length, query channels

        q = (self.q_proj(q).view(Bq, Tq, self.n_heads, Cq//self.n_heads).transpose(1, 2))
        k = (self.k_proj(k).view(Bk, Tk, self.n_heads, Ck//self.n_heads).transpose(1, 2))
        v = (self.v_proj(v).view(Bk, Tk, self.n_heads, Ck//self.n_heads).transpose(1, 2))
        # q, k, v: (batch_size, n_heads, seq_len, d_model/n_heads)

        # attn_mask: (B, Tk) -> (B, 1, 1, Tk) -> (B, n_heads, Tq, Tk)
        attn_mask = attn_mask.to(q.dtype) 
        attn_mask = attn_mask.view(Bk, 1, 1, Tk).expand(-1, self.n_heads, Tq, -1)

        attn_output = stable_scaled_dot_product_attention2(q, k, v, attn_mask, mask_future=mask_future, dropout=self.attn_drop)
        # attn_output: (B, n_heads, Tq, Tk) @ (B, n_heads, Tk, Ck//n_heads) -> (B, n_heads, Tq, Ck//n_heads)

        # re-assemble all head outputs side by side
        B = max(Bk, Bq)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Tq, Cq)
        # attn_output: (batch_size, seq_len, d_model)
        return self.resid_drop(self.out_proj(attn_output))
    
class FastMultiHeadAttention(nn.Module):
    """
    Fast Multi Head Attention:
    Uses one qkv projection for all heads, and 3 dim matrix multiplication for faster computation.
    Only works correctly if q, k, v are of same size.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f'd_model ({d_model}) must be divisible by n_heads ({n_heads}) without remainder'
        self.n_heads = n_heads

        # key, query, value projections stacked for all heads, more efficient
        # con: q, k, v need to be of same size
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # regularization
        self.resid_drop = nn.Dropout(dropout)
        self.attn_drop = dropout

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, mask_future: bool = False) -> torch.Tensor:
        assert q.size() == k.size() == v.size(), f'q, k, v must have the same shape, got {q.size()}, {k.size()}, {v.size()}'
        B, T, C = q.size() # batch size, key sequence length, key channels

        # (batch size, sequence length, d_model) -> (batch size, sequence length, 3*d_model)
        qkv = self.qkv_proj(torch.cat([q, k, v], dim=-1))
        # (batch size, sequence length, 3*d_model) -> (batch size, sequence length, n_heads, 3*d_model//n_heads)
        # -> (batch size * n_heads, sequence length, 3*d_model//n_heads), enables 3 dim matrix multiplication -> faster/more efficient
        qkv = qkv.view(B, T, self.n_heads, 3 * C//self.n_heads).transpose(1, 2).view(B * self.n_heads, T, 3 * C//self.n_heads)
        # split in q, k, v
        q, k, v = qkv.chunk(3, dim=-1)

        # enables simpler masking procedure
        attn_mask = attn_mask.to(q.dtype)
        # On axis 0, copy the first item (scalar or vector) for n_heads times, then copy the next item, and so on
        attn_mask = torch.repeat_interleave(attn_mask, repeats=self.n_heads, dim=0)

        # (batch size * n_heads, sequence length, sequence length) @ (batch size * n_heads, sequence length, d_model//n_heads) 
        # -> (B * n_heads, Tq, Ck//n_heads)
        attn_output = stable_scaled_dot_product_attention2(q, k, v, attn_mask, mask_future=mask_future, dropout=self.attn_drop)

        # re-assemble all head outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        # attn_output: (batch_size, seq_len, d_model)
        return self.resid_drop(self.out_proj(attn_output))
