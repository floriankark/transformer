# Implement the word embedding layer in pytorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WordEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, padding_idx: int = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 2, f'Expected: (batch size, max sequence length), got {input.shape}'
        return self.embedding(input) * math.sqrt(self.d_model)
