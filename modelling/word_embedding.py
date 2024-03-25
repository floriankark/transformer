import torch
import torch.nn as nn
from math import sqrt

class Embedding(nn.Module):
    """
    Embedding layer for the transformer model. It is a simple lookup table that multiplies the input by sqrt(d_model).
    """
    def __init__(
        self, vocab_size: int, d_model: int) -> None:
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 2, f'Expected: (batch size, max sequence length), got {input.shape}'
        return self.lut(input) * sqrt(self.d_model)
