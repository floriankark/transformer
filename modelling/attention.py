import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        return F.softmax(q @ k.T / np.sqrt(q.size(-1)) + mask, dim=-1) @ v
