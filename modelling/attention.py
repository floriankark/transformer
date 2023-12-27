import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, q, k, v, attention_mask=None):
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
