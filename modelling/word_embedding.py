import torch
import torch.nn as nn
import math

class WordEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int) -> None:
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 2, f'Expected: (batch size, max sequence length), got {input.shape}'
        return self.lut(input) * math.sqrt(self.d_model)
