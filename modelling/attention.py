import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super().__init__()

    def forward(self, q, k, v):
        mask = (
            -1e10 * torch.triu(torch.ones(q.size(1), k.size(1)), diagonal=1)
            if self.mask
            else 0
        )  # upper triangular matrix [n_q, n_k] -∞
        return F.softmax(q @ k.T / np.sqrt(q.size(-1)) + mask, dim=-1) @ v
